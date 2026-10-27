"""从 PMGen 提取的结构建模实现；上游来源见 SOURCE_MANIFEST.json。"""
import ast
import concurrent.futures
import contextlib
import glob
import json
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import re
import shutil
import subprocess
import sys
import warnings
import numpy as np
import pandas as pd
import psutil
from PANDORA import Target, Pandora, Database
from utils import processing_functions
warnings.filterwarnings('ignore', category=FutureWarning)

class StructureModeler():
    def __init__(self, peptide, mhc_seq, mhc_type, id, output_dir='output',
                  anchors=None, mhc_allele=None, predict_anchor=True,
                 num_templates=4, num_recycles=3, models=['model_2_ptm'],
                 alphafold_param_folder = 'AFfine/af_params/params_original/',
                 fine_tuned_model_path='AFfine/af_params/params_finetune/params/model_ft_mhc_20640.pkl',
                 benchmark=False, n_homology_models=1, best_n_templates=4,
                 pandora_force_run=True, no_modelling=False,
                 return_all_outputs=False, 
                 benchmark_similarity_threshold=0.95, benchmark_exclude_ids=None, # added after review --> similarity threshold
                sampling_mode=False, n_times_sampling=200, sampling_fraction_IG=0.5, # Sampling mode added
                sampling_fraction_evo=0.3, sampling_dropout_rate=0.5, sampling_seed=42,
                radius=8.0, pep_sampling=None, benchmark_release_dates_map=None, 
                target_release_date=None,
    ):
        """
        Initializes the PMGen modeling pipeline.

        Args:
            peptide (str): Peptide sequence.
            mhc_seq (str): MHC sequence(s) (single for MHC-I, two chains for MHC-II).
            mhc_type (int): MHC class (1 or 2).
            id (str): Unique identifier for the run.
            output_dir (str): Directory for output files. Default is 'output'.
            anchors (list or tuple, optional): Anchor positions for MHC binding.
            mhc_allele (str, optional): Specific MHC allele name.
            predict_anchor (bool): Whether to predict anchor residues. Default is True.
            num_templates (int): Number of templates used in modeling.
            num_recycles (int): Number of AlphaFold recycling iterations.
            models (list): List of AlphaFold models to use.
            alphafold_param_folder (str): Path to AlphaFold parameter files.
            fine_tuned_model_path (str): Path to fine-tuned AlphaFold model.
            benchmark (bool): Use different allele compared to the actual allele. make sure the id shouldbe pdb id.
            n_homology_models (int): number of initial peptide homology models to generate by modeller, default=1.
            best_n_templates (int): number of found templates used for homology modeling via modeler, default=4.
            pandora_force_run (bool): Weather to force run pandora or not, default=True.
            no_modelling (bool): If active, no modeller homology modeling happens and only PANDORA is used for template search and alignment.
            return_all_outputs (bool): If active, all alphafold outputs are saved.
            benchmark_similarity_threshold (float): Only used during benchmarking, exludes sequences above this similarity threshold.
            benchmark_exclude_ids (bool): Exclude benchmark ids in given df in run_PMGen.py
        """
        super().__init__()
        self.peptide = peptide
        self.mhc_seq = mhc_seq
        self.mhc_type = mhc_type
        self.output_dir = output_dir
        self.anchors = anchors
        self.mhc_allele = mhc_allele
        self.predict_anchor = predict_anchor
        self.id = id
        self.num_templates = num_templates
        self.num_recycles = num_recycles
        self.models = models
        self.alphafold_param_folder = alphafold_param_folder
        self.fine_tuned_model_path = fine_tuned_model_path
        self.benchmark = benchmark
        self.n_homology_models = n_homology_models
        self.best_n_templates = best_n_templates
        self.pandora_force_run = pandora_force_run
        self.no_modelling = no_modelling
        self.return_all_outputs = return_all_outputs
        self.benchmark_similarity_threshold = benchmark_similarity_threshold # added after review --> similarity threshold
        self.benchmark_exclude_ids = benchmark_exclude_ids # added after review --> similarity threshold
        self.benchmark_release_dates_map = benchmark_release_dates_map   # NEW
        self.target_release_date = target_release_date                   # NEW
        self.sampling_mode = sampling_mode
        self.n_times_sampling = n_times_sampling
        self.sampling_fraction_IG = sampling_fraction_IG
        self.sampling_fraction_evo = sampling_fraction_evo
        self.sampling_dropout_rate = sampling_dropout_rate
        self.sampling_seed = sampling_seed
        self.radius = radius
        self.pep_sampling = pep_sampling
        
        self.input_assertion()
        if len(self.models) > 1:
            print(f'\n #### Warning! You are running for multiple models {self.models}'
                  f'Please make sure your model names are correct.'
                  f'For fine-tuned models please use "_ft" as a identifier in model param '
                  f'.pkl file ####\n')
        # input derived args
        self.mhc_type_greek='I' if self.mhc_type==1 else 'II'
        self.m_chain, self.n_chain = (self.mhc_seq + '/').split('/')[0], (self.mhc_seq + '/').split('/')[1]
        self.pandora_output = os.path.join(self.output_dir, 'pandora')
        self.db = Database.load() # load pandora db
        self.alignment_output = os.path.join(self.output_dir, 'alignment')
        self.alphafold_out = os.path.join(self.output_dir, 'alphafold')
        self.alphafold_input_file = os.path.join(self.alphafold_out, self.id, f'alphafold_input_file.tsv')
        # vars defined later
        self.template_id = None
        self.no_modelling_output_dict = None

    def run_PMGen(self, test_mode=False, run_alphafold=True):
        """
        Runs the full PMGen pipeline, including Pandora alignment and AlphaFold prediction.
        Args:
            test_mode (bool): If True, runs in test mode without full execution.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        self.template_id = self.run_pandora(self.pandora_force_run)
        #pandora_template_path = os.path.join(self.pandora_output, self.id, self.template_id)
        #aln_output_file = os.path.join(self.alignment_output, self.id + 'no_pep.tsv')
        ## create aln files for non-pep
        #df_aln_nopep = self.alignment_without_peptide(self.template_id, aln_output_file, pandora_template_path)
        ## create aln files for with-pep
        # list of pandora generated template paths
        pdb_files = processing_functions.rename_files(os.path.join(self.pandora_output, self.id), self.num_templates)
        mhc_pep_seq = self.mhc_seq + '/' + self.peptide
        aln_output_file = os.path.join(self.alignment_output, self.id + '_with_pep.tsv')
        if not self.no_modelling:
            _ = self.alignment_with_peptide(pdb_files, mhc_pep_seq, output_path=aln_output_file)
        else:
            os.makedirs(self.alignment_output, exist_ok=True)
            processing_functions.alignment_to_df(self.no_modelling_output_dict, output_dir=aln_output_file)
        ## Prepare Alphafold Fine Input files
        os.makedirs(self.alphafold_out + f'/{self.id}', exist_ok=True)
        self.alphafold_preparation(template_aln_file=aln_output_file, mhc_pep_seq=mhc_pep_seq, output=self.alphafold_input_file)
        self.output_pdbs_dict = {}
        if run_alphafold:
            print('## To run Alphafold Please Make Sure GPU is Available and can be found ##')
            self.run_alphafold(input_file=self.alphafold_input_file, output_prefix=self.alphafold_out + f'/{self.id}/')
            # get the paths for proteinmpnn
            self.output_pdbs_dict[self.id] = [os.path.join(self.alphafold_out, i) for i in os.listdir(self.alphafold_out) if i.endswith('.pdb') and 'model_' in i and not i.endswith('.npy')]

    def run_pandora(self, force_run=True):
        """
        Runs the Pandora module to generate template structures for MHC modeling.

        Returns:
            str: Template ID used in the modeling process.
        """
        
        os.makedirs(self.pandora_output, exist_ok=True)
        log_file = f'{self.pandora_output}/{self.id}/pandora.log'
        mhc_allele = [] if not self.mhc_allele else [self.mhc_allele]
        anchor = [] if self.anchors is None else self.anchors
        # Redirect stdout and stderr to the log file
        os.makedirs(self.pandora_output + '/' + self.id, exist_ok=True)
        shoud_I_run = 'Yes'
        if not force_run:
            # check if mod*.pdb and template file exists
            try:
                files = [file for file in glob.glob(os.path.join(self.pandora_output, self.id, '????.pdb')) if
                         "mod" not in file.split("/")[-1]]
                if files:
                    template_id = files[0].split("/")[-1]
                models = [file for file in glob.glob(os.path.join(self.pandora_output, self.id, '????.pdb')) if
                         "mod" in file.split("/")[-1]]
                if len(models) >= self.num_templates:
                    shoud_I_run = 'No'
                print(f'Mode force_run == {force_run}, and PANDORA has already finished for {self.id}, no need to run it!')
            except:
                shoud_I_run = 'Yes'

        if shoud_I_run == 'Yes':
            with open(log_file, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                try:
                    print(f"Starting {self.id} initialization...")
                    target = Target(id=self.id, peptide=self.peptide, allele_type=mhc_allele,
                                    MHC_class=self.mhc_type_greek, M_chain_seq=self.m_chain,
                                    N_chain_seq=self.n_chain, output_dir=self.pandora_output,
                                    use_netmhcpan=self.predict_anchor, anchors=anchor)
                    case = Pandora.Pandora(target, self.db, no_modelling=self.no_modelling)
                    case.model(n_loop_models=self.num_templates, benchmark=self.benchmark,
                            benchmark_similarity_threshold=(self.benchmark_similarity_threshold if self.benchmark else None), # added after review --> similarity threshold
                            benchmark_exclude_ids=(self.benchmark_exclude_ids if self.benchmark else None),
                            benchmark_release_dates_map=(self.benchmark_release_dates_map if self.benchmark else None),
                            target_release_date=(self.target_release_date if self.benchmark else None),    
                            n_homology_models=self.n_homology_models,
                            best_n_templates=self.best_n_templates) 
                    print("Pandora modeling completed successfully.")
                    # Persist similarity info for later CSV aggregation # added after review --> similarity threshold
                    if self.benchmark and getattr(case, 'similarity_info', None) is not None:
                        # Source of truth: no_modelling_output_dict['template_id'] (all templates actually used).
                        # Fall back to case.template if no_modelling_output_dict isn't populated.
                        nmod = getattr(case, 'no_modelling_output_dict', {}) or {}
                        selected_ids = list(nmod.get('template_id') or [])
                        if not selected_ids:
                            tmpl_attr = case.template
                            templates_list = tmpl_attr if isinstance(tmpl_attr, (list, tuple)) else [tmpl_attr]
                            selected_ids = [t.id for t in templates_list]

                        sims = case.similarity_info.get('all_similarities', {})
                        template_sims = [(tid, sims.get(tid[:4], None)) for tid in selected_ids]
                        info = {
                            'id': self.id,
                            'min_similarity': case.similarity_info['min_similarity'],
                            'at_least_one_below_threshold': case.similarity_info['at_least_one_below_threshold'],
                            'n_templates_used': len(selected_ids),
                            'template_similarities': template_sims,
                            'n_templates_below_threshold': case.similarity_info['n_templates_below_threshold'],
                        }
                        json_path = os.path.join(self.pandora_output, self.id, 'benchmark_similarity.json')
                        with open(json_path, 'w') as fjs:
                            json.dump(info, fjs)
                except Exception as e:
                    print(f"❌ An error occurred during template engineering {self.id}: {str(e)}", file=sys.stderr)
                    raise
        print("✔Pandora run completed. Check log file for details:", log_file)
        if self.no_modelling:
            if shoud_I_run == 'Yes':
                self.no_modelling_output_dict = case.no_modelling_output_dict

            else:
                with open(os.path.join(self.pandora_output, self.id, 'no_modelling_output_dict.json'), 'r') as f:
                    self.no_modelling_output_dict = json.load(f)
            self.no_modelling_assertion()

        # get template id used in pandora
        if not self.no_modelling:
            files = [file for file in glob.glob(os.path.join(self.pandora_output, self.id, '????.pdb')) if
                     "mod" not in file.split("/")[-1]]
            if files:
                template_id = files[0].split("/")[-1]
                print(f"✔ {self.id} log: Template ID used for homology modeling: {template_id}")
                return template_id
            else:
                print(f"❌ {self.id} log: No template ID found.")
                return None
        else: print(f"No modelling mode is done, files generated in: {self.pandora_output + '/' + self.id}")

    def alignment_without_peptide(self, template_id, output_path, template_path,
                                       template_csv_path=str(Path(__file__).resolve().parents[1] / "data/all_templates.csv")):
        """
        Generates an alignment file for MHC without the peptide.
        Args:
            template_id (str): PDB ID used as a template.
            output_path (str): File path to save the alignment.
            template_path (str): Path to the template structure.
            template_csv_path (str): Path to the CSV file containing template sequences.
        Returns:
            pd.DataFrame: Alignment data.
        """
        os.makedirs(self.alignment_output, exist_ok=True)
        df = processing_functions.prepare_alignment_file_without_peptide(template_id=template_id,
                                                                    mhc_seq=self.mhc_seq,
                                                                    mhc_type=self.mhc_type,
                                                                    output_path=output_path,
                                                                    template_path=template_path,
                                                                    template_csv_path=template_csv_path)
        return df

    def alignment_with_peptide(self, pdb_files, mhc_pep_seq, output_path):
        """
        Generates an alignment file for MHC with the peptide.

        Args:
            pdb_files (list): List of PDB template file paths.
            mhc_pep_seq (str): Full MHC-peptide sequence.
            output_path (str): File path to save the alignment.

        Returns:
            pd.DataFrame: Alignment data.
        """
        os.makedirs(self.alignment_output, exist_ok=True)
        DF = []
        for pdb_file in pdb_files:
            df = processing_functions.prepare_alignment_file_with_peptide(pdb_file=pdb_file,
                                                                     target_seq=mhc_pep_seq,
                                                                     mhc_type=self.mhc_type,
                                                                     output_path=None,
                                                                     peptide=True)
            DF.append(df)
        DF = pd.concat(DF, ignore_index=True)
        DF.to_csv(output_path, sep='\t', index=False)
        return DF

    def alphafold_preparation(self, template_aln_file, mhc_pep_seq, output):
        """
        Prepares input files for AlphaFold.

        Args:
            template_aln_file (str): Path to the template alignment file.
            mhc_pep_seq (str): Full MHC-peptide sequence.
            output (str): Path to save the prepared input file.
        """
        template_pdb_dict_path = os.path.join(self.pandora_output, self.id, "no_modelling_output_dict.json")
        df = pd.DataFrame({"target_chainseq": [mhc_pep_seq],
                           "templates_alignfile": [template_aln_file],
                           "targetid": [self.id],
                           "template_pdb_dict": [template_pdb_dict_path]})
        df.to_csv(output, sep='\t', index=False)

    def run_alphafold(self, input_file, output_prefix):
        """
        Runs AlphaFold with the specified input and parameters.

        Args:
            input_file (str): Path to the input file.
            output_prefix (str): Prefix for output files.
        """
        model_params_files = ''
        model_names = ''
        for model in self.models:
            i = 'classic' if '_ft' not in model else self.fine_tuned_model_path
            model_params_files += f'{i} '
            model_names += f'{model} '

        model_names = model_names.rstrip()
        model_params_files = model_params_files.rstrip()
        command = [
            sys.executable, "AFfine/run_prediction.py",
            "--targets", f"{input_file}",
            "--data_dir", f"{self.alphafold_param_folder}",
            "--outfile_prefix", f"{output_prefix}",
            "--model_names", *model_names.split(),
            "--model_params_files", *model_params_files.split(),
            "--ignore_identities",
            "--num_recycles", f"{self.num_recycles}", 
            "--num_templates", f"{self.num_templates}"
        ]
        if self.sampling_mode:
            command += ['--sampling_mode',
                        '--return_all_outputs',
                        '--n_times_sampling', str(self.n_times_sampling),
                        '--sampling_fraction_IG', str(self.sampling_fraction_IG),
                        '--sampling_fraction_evo', str(self.sampling_fraction_evo),
                        '--sampling_dropout_rate', str(self.sampling_dropout_rate),
                        '--sampling_seed', str(self.sampling_seed),
                        '--radius', str(self.radius)]
            if self.pep_sampling:
                command += ['--pep_sampling', self.pep_sampling]
        if not self.no_modelling:
            command += ['--no_initial_guess']
        else:
            print(' -- Alphafold Initial Guess Mode, No homology models will be used --')
        if self.return_all_outputs:
            command += ['--return_all_outputs']

        print('AFfine Command: \n',command)
        print("### Running AFfine/run_prediction.py ###", flush=True)
        try:
            # Inherit both streams so large diagnostics cannot block a pipe.
            subprocess.run(command, check=True)
        except Exception:
            self.error_handling()
            raise
        print("\n✔ Alphafold executed successfully!", flush=True)

    def error_handling(self, errotype='affine'):
        if errotype == 'affine':
            print('### ERROR MESSAGE ###:'
                  'Alphafold Run Failed, It is common, please follow the debugging steps below: \n'
                  '1- check if you have ptxas: "which ptxas" if not do:\n'
                  'conda install -c nvidia cuda-nvcc\n'
                  'or\n'
                  'export PATH=/usr/local/cuda/bin:$PATH\n'
                  'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH')

    def no_modelling_assertion(self):

        required_keys = [
            'template_anchors', 'target_anchors', 'template_id', 'aln_M',
            'aln_B_N', 'aln_P', 'aln_template', 'aln_target', 'template_path'
        ]
        missing_keys = [key for key in required_keys if key not in self.no_modelling_output_dict]
        assert not missing_keys, f"Missing keys in no_modelling_output_dict: {missing_keys}"
        empty_keys = [key for key in required_keys if not self.no_modelling_output_dict[key]]
        assert not empty_keys, f"Empty values found for keys: {empty_keys}"

    def input_assertion(self):
        assert isinstance(self.peptide, str), f"peptide must be a string, found: {self.peptide}"
        assert self.mhc_type in [1, 2], f"mhc_type must be an integer value of 1 or 2, found: {self.peptide}"
        assert isinstance(self.mhc_seq, str), f"mhc_seq must be a string, found: {self.mhc_seq}"
        assert isinstance(self.output_dir, str), f"output_dir must be a string, found: {self.output_dir}"
        if self.anchors:
            assert isinstance(self.anchors, (tuple, list)), (f"anchors must be a tuple or list, found: {self.anchors}"
                                                             f"alternatively use predict_anchor==True")
            self.anchors = list(self.anchors)
        else:
            assert self.predict_anchor==True, f'If anchors arg is empty, please set predict_anchor=True'
        if self.mhc_allele is not None:
            assert isinstance(self.mhc_allele, str), (f'mhc_allele should be string, found: {self.mhc_allele}'
                                                      f'\n example for MHC-I: HLA-B40:02'
                                                      f'\n example for MHC-II: HLA-DRA01;HLA-DRB11')
        assert isinstance(self.predict_anchor, bool), f'predict_anchor must be a bool, found:{self.predict_anchor}'
        if self.mhc_type==2:
            assert len(self.mhc_seq.split('/')) == 2, (f'mhc_seq for mhc_type==2 should be two '
                                             f'sequences separated by "/", found: {self.mhc_seq}'
                                             f'\n please follow the order "M-chain/Nchain"')
        elif self.mhc_type==1:
            assert len(self.mhc_seq.split('/')) == 1, (f'mhc_seq for mhc_type==1 should be one '
                                             f'sequence without "/" as seperator: {self.mhc_seq}')
        assert isinstance(self.id, str), f'id must be a string, found: {self.id}'
        assert isinstance(self.num_templates, int), f'num_templates must be an integer, found {self.num_templates}'
        assert isinstance(self.num_recycles, int), f'num_recycles must be an integer, found {self.num_recycles}'
        assert isinstance(self.models, list), f'models must be a list, found {self.models}'
        assert isinstance(self.fine_tuned_model_path, str), f'fine_tuned_model_path must be a string, found {self.fine_tuned_model_path}'
        assert isinstance(self.alphafold_param_folder, str), f'alphafold_param_folder must be a string, found {self.alphafold_param_folder}'


class StructureBatch():
    def __init__(self, df, output_dir, num_templates=4, num_recycles=3, models=['model_2_ptm'],
                 alphafold_param_folder='AFfine/af_params/params_original/',
                 fine_tuned_model_path='AFfine/af_params/params_finetune/params/model_ft_mhc_20640.pkl',
                 max_ram_per_job=3, num_cpu=1, benchmark=False, best_n_templates=1, n_homology_models=1,
                 pandora_force_run=True, no_modelling=False, return_all_outputs=False, 
                 benchmark_similarity_threshold=0.95, benchmark_exclude_ids=None,  # added after review --> similarity threshold
                 sampling_mode=False, n_times_sampling=200, sampling_fraction_IG=0.5, # Sampling mode added
                sampling_fraction_evo=0.3, sampling_dropout_rate=0.5, sampling_seed=42,
                radius=8.0, pep_sampling=None, benchmark_release_dates_map=None,
                 ):
        """
        Initializes the run_PMGen_wrapper class.
        :param df: pandas DataFrame containing input data. Required columns:
            - 'peptide' (str): Peptide sequence.
            - 'mhc_seq' (str): MHC sequence (one chain for MHC-I, two for MHC-II).
            - 'mhc_type' (int): Type of MHC (1 for MHC-I, 2 for MHC-II).
            - 'anchors' (str or NaN): Two numbers (MHC-I) or four numbers (MHC-II) separated by ";". If not provided, anchors will be predicted.
            - 'id' (str): Unique identifier for each row.
        :param output_dir: str, path to the output directory. This directory will be created if it does not exist.
        :param num_templates: int, number of structural templates to use (default=4).
        :param num_recycles: int, number of recycles in AlphaFold inference (default=3).
        :param models: list of str, names of AlphaFold models to use (default=['model_2_ptm']).
        :param alphafold_param_folder: str, path to the folder containing original AlphaFold model parameters.
            - Must be an existing directory.
        :param fine_tuned_model_path: str, path to the fine-tuned AlphaFold model parameters.
            - Must be an existing file.
        :param max_ram_per_job: int, maximum RAM (in GB) per parallel process (default=3).
        :param num_cpu: int, number of CPU cores to use (default=1).
        :param benchmark: bool, to do becnhmarking.
        :param best_n_templates: int, how many models to used for homology modeling after sequence aln search, default=1.
        :param n_homology_models: int, number of initial models to be done with modeller homology modelling, default=1.
        :param no_modelling (bool): If active, no modeller homology modeling happens and only PANDORA is used for template search and alignment.
        :param pandora_force_run (bool): If active, PANDORA will be forced to run even if files already exist.
        :param return_all_outputs (bool): If active, all alphafold outputs are saved.
        :param benchmark_similarity_threshold (float): Only used during benchmarking, exludes sequences above this similarity threshold.
        :param benchmark_release_dates_map: Only valid with --benchmark. When set, templates whose PDB release 
                date is >= the target structure\'s release date are excluded. Requires a "release_date" (or "PDB release date") column in --df. 
                Templates whose PDB IDs are not in --df are kept (assumed pre-2018).
        The function `input_assertion()` checks if all inputs are correctly formatted and whether required files and directories exist.

        Raises:
            - AssertionError if any input is invalid.
        """
        self.df = df
        self.output_dir = output_dir
        self.num_templates = num_templates
        self.num_recycles = num_recycles
        self.models = models
        self.alphafold_param_folder = alphafold_param_folder
        self.fine_tuned_model_path = fine_tuned_model_path
        self.max_ram_per_job = max_ram_per_job
        self.num_cpu = num_cpu
        self.benchmark = benchmark
        self.best_n_templates = best_n_templates
        self.n_homology_models = n_homology_models
        self.pandora_force_run = pandora_force_run
        self.no_modelling = no_modelling
        self.return_all_outputs = return_all_outputs
        self.benchmark_similarity_threshold = benchmark_similarity_threshold # added after review --> similarity threshold
        self.benchmark_exclude_ids = benchmark_exclude_ids
        self.benchmark_release_dates_map = benchmark_release_dates_map 
        self.sampling_mode = sampling_mode
        self.n_times_sampling = n_times_sampling
        self.sampling_fraction_IG = sampling_fraction_IG
        self.sampling_fraction_evo = sampling_fraction_evo
        self.sampling_dropout_rate = sampling_dropout_rate
        self.sampling_seed = sampling_seed
        self.radius = radius
        self.pep_sampling = pep_sampling
        self.input_assertion()

    def run_wrapper(self, run_alphafold=True):
        INPUT_DF = []
        for step, row in self.df.iterrows():
            anchors = [int(r) for r in row['anchors'].split(';')] if isinstance(row['anchors'], str) and ';' in row['anchors'] else None
            try:
                mhc_allele = row['mhc_allele'] if row['mhc_allele'] else None
            except:
                mhc_allele = None
            predict_anchor = False if anchors else True
            runner = StructureModeler(peptide=row['peptide'], mhc_seq=row['mhc_seq'],
                                           mhc_type=row['mhc_type'], id=f"{row['id']}", output_dir=self.output_dir,
                                            anchors=anchors, mhc_allele=mhc_allele, predict_anchor=predict_anchor,
                                            num_templates=self.num_templates, num_recycles=self.num_recycles,
                                            models=self.models, alphafold_param_folder=self.alphafold_param_folder,
                                            fine_tuned_model_path=self.fine_tuned_model_path, benchmark=self.benchmark,
                                            n_homology_models=self.n_homology_models, best_n_templates=self.best_n_templates,
                                            pandora_force_run=self.pandora_force_run, no_modelling=self.no_modelling,
                                            return_all_outputs=self.return_all_outputs, 
                                            benchmark_similarity_threshold=self.benchmark_similarity_threshold, benchmark_exclude_ids=self.benchmark_exclude_ids, # added after review --> similarity threshold
                                            benchmark_release_dates_map=self.benchmark_release_dates_map,
                                            target_release_date=(
                                                self.benchmark_release_dates_map.get(str(row['id']).split('_')[0][:4].upper())
                                                if self.benchmark_release_dates_map is not None else None
                                            ),
                                            sampling_mode=self.sampling_mode, #v2
                                            n_times_sampling=self.n_times_sampling,
                                            sampling_fraction_IG=self.sampling_fraction_IG,
                                            sampling_fraction_evo=self.sampling_fraction_evo,
                                            sampling_dropout_rate=self.sampling_dropout_rate,
                                            sampling_seed=self.sampling_seed,
                                            radius=self.radius,
                                            pep_sampling=self.pep_sampling,
                                        )
            runner.run_PMGen(test_mode=False, run_alphafold=False)
            input_df = pd.read_csv(runner.alphafold_input_file, sep='\t', header=0)
            input_df['targetid'] = [str(row['id']) + '/' + str(row['id'])] # id/id
            INPUT_DF.append(input_df)
        self._aggregate_benchmark_similarity_csv() # added after review --> similarity threshold
        if run_alphafold:
            alphafold_out = self.output_dir + '/alphafold'
            pd.concat(INPUT_DF).to_csv(f'{alphafold_out}/alphafold_input_file.tsv', sep='\t', index=False)
            runner.run_alphafold(input_file=f'{alphafold_out}/alphafold_input_file.tsv', output_prefix=alphafold_out + '/')
        


    def get_available_memory(self):
        """ Returns available system memory in GB """
        memory = psutil.virtual_memory()
        return memory.available / (1024 ** 3)  # Convert bytes to GB


    def process_row(self, row):
        """ Process each row to generate input data for Alphafold """
        anchors = [int(r) for r in row['anchors'].split(';')] if isinstance(row['anchors'], str) and ';' in row[
            'anchors'] else None
        try:
            mhc_allele = row['mhc_allele'] if row['mhc_allele'] else None
        except:
            mhc_allele = None
        predict_anchor = False if anchors else True
        runner = StructureModeler(peptide=row['peptide'], mhc_seq=row['mhc_seq'],
                                        mhc_type=row['mhc_type'], id=f"{row['id']}", output_dir=self.output_dir,
                                        anchors=anchors, mhc_allele=mhc_allele, predict_anchor=predict_anchor,
                                        num_templates=self.num_templates, num_recycles=self.num_recycles,
                                        models=self.models, alphafold_param_folder=self.alphafold_param_folder,
                                        fine_tuned_model_path=self.fine_tuned_model_path, benchmark=self.benchmark,
                                        n_homology_models=self.n_homology_models, best_n_templates=self.best_n_templates,
                                        pandora_force_run=self.pandora_force_run, no_modelling=self.no_modelling,
                                        return_all_outputs=self.return_all_outputs, 
                                        benchmark_similarity_threshold=self.benchmark_similarity_threshold, benchmark_exclude_ids=self.benchmark_exclude_ids, # added after review --> similarity threshold
                                        benchmark_release_dates_map=self.benchmark_release_dates_map,
                                        target_release_date=(
                                            self.benchmark_release_dates_map.get(str(row['id']).split('_')[0][:4].upper())
                                            if self.benchmark_release_dates_map is not None else None
                                        ),
                                        sampling_mode=self.sampling_mode, #v2
                                        n_times_sampling=self.n_times_sampling,
                                        sampling_fraction_IG=self.sampling_fraction_IG,
                                        sampling_fraction_evo=self.sampling_fraction_evo,
                                        sampling_dropout_rate=self.sampling_dropout_rate,
                                        sampling_seed=self.sampling_seed,
                                        radius=self.radius,
                                        pep_sampling=self.pep_sampling,
        )
        runner.run_PMGen(run_alphafold=False)
        input_df = pd.read_csv(runner.alphafold_input_file, sep='\t', header=0)
        input_df['targetid'] = [str(row['id']) + '/' + str(row['id'])]  # id/id
        return input_df

    def run_wrapper_parallel(self, max_ram=3, max_cores=4, run_alphafold=True):
        """
        Processes rows of input data in parallel, utilizing available system memory and CPU cores.
        It ensures that the system memory does not exceed the specified `max_ram` per job,
        divides the work among multiple processes, and then runs Alphafold on the final input file.
        Args:
            max_ram (float, optional): Maximum amount of system memory (in GB) allocated per parallel job. Default is 3 GB.
            max_cores (int, optional): Maximum number of CPU cores to use for parallel processing. Default is 4.
        """
        # List to hold the processed dataframes
        INPUT_DF = []
        # Monitor system memory to ensure it doesn't exceed max_ram
        available_memory = self.get_available_memory()
        print(f"Available memory: {available_memory} GB")
        # Calculate maximum number of jobs based on available memory and max_ram (in GB)
        max_jobs = max_cores
        if available_memory < max_ram:
            max_jobs = int(np.floor(available_memory / max_ram))  # Limit jobs based on available memory
        print(f"Max concurrent jobs based on available memory: {max_jobs}")
        # Use ProcessPoolExecutor for parallelism
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_jobs) as executor:
            # Submit tasks for each row
            futures = {executor.submit(self.process_row, row): row for _, row in self.df.iterrows()}
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    INPUT_DF.append(result)
                except Exception as e:
                    print(f"Error processing row: {e}")
        # Combine all the dataframes into one and save to file
        alphafold_out = self.output_dir + '/alphafold'
        pd.concat(INPUT_DF).to_csv(f'{alphafold_out}/alphafold_input_file.tsv', sep='\t', index=False)
        # Now, run alphafold on the final input file
        # Initialize a final runner for Alphafold
        # Run alphafold model, initialize a runner
        row = self.df.iloc[0, :]
        anchors = [int(r) for r in row['anchors'].split(';')] if isinstance(row['anchors'], str) and ';' in row[
            'anchors'] else None
        try:
            mhc_allele = row['mhc_allele'] if row['mhc_allele'] else None
        except:
            mhc_allele = None
        predict_anchor = False if anchors else True
        runner = StructureModeler(peptide=row['peptide'], mhc_seq=row['mhc_seq'],
                                            mhc_type=row['mhc_type'], id=f"{row['id']}", output_dir=self.output_dir,
                                            anchors=anchors, mhc_allele=mhc_allele, predict_anchor=predict_anchor,
                                            num_templates=self.num_templates, best_n_templates=self.best_n_templates,
                                            num_recycles=self.num_recycles,
                                            models=self.models, alphafold_param_folder=self.alphafold_param_folder,
                                            fine_tuned_model_path=self.fine_tuned_model_path, no_modelling=self.no_modelling,
                                            return_all_outputs=self.return_all_outputs,
                                            benchmark_release_dates_map=self.benchmark_release_dates_map,
                                            target_release_date=(
                                                self.benchmark_release_dates_map.get(str(row['id']).split('_')[0][:4].upper())
                                                if self.benchmark_release_dates_map is not None else None
                                            ),
                                            sampling_mode=self.sampling_mode, #v2
                                            n_times_sampling=self.n_times_sampling,
                                            sampling_fraction_IG=self.sampling_fraction_IG,
                                            sampling_fraction_evo=self.sampling_fraction_evo,
                                            sampling_dropout_rate=self.sampling_dropout_rate,
                                            sampling_seed=self.sampling_seed,
                                            radius=self.radius,
                                            pep_sampling=self.pep_sampling,)
        self._aggregate_benchmark_similarity_csv() # added after review --> similarity threshold
        if run_alphafold:
            runner.run_alphafold(input_file=f'{alphafold_out}/alphafold_input_file.tsv', output_prefix=alphafold_out + '/')
        
    def _aggregate_benchmark_similarity_csv(self):
        if not self.benchmark:
            return
        rows = []
        max_t = 0
        for _, row in self.df.iterrows():
            json_path = os.path.join(self.output_dir, 'pandora', str(row['id']), 'benchmark_similarity.json')
            if not os.path.exists(json_path):
                continue
            with open(json_path) as fjs:
                info = json.load(fjs)
            max_t = max(max_t, len(info['template_similarities']))
            rows.append(info)
        if not rows:
            return
        out_rows = []
        for info in rows:
            d = {
                'id': info['id'],
                'min_similarity': info['min_similarity'],
                'at_least_one_below_threshold': info['at_least_one_below_threshold'],
                'n_templates_used': info['n_templates_used'],
                "n_templates_below_threshold": info["n_templates_below_threshold"]
            }
            for i in range(max_t):
                if i < len(info['template_similarities']):
                    tid, sim = info['template_similarities'][i]
                    d[f'template_{i+1}_id'] = tid
                    d[f'template_{i+1}_similarity'] = sim
                else:
                    d[f'template_{i+1}_id'] = None
                    d[f'template_{i+1}_similarity'] = None
            out_rows.append(d)
        pd.DataFrame(out_rows).to_csv(os.path.join(self.output_dir, 'benchmark_similarity.csv'), index=False)

    def input_assertion(self):
        """
        Validates input arguments to ensure correct data types and formats.
        """
        assert isinstance(self.df, pd.DataFrame), f"df must be a pandas DataFrame, found: {type(self.df)}"
        assert isinstance(self.output_dir,
                          str), f"output_dir must be a string (directory path), found: {type(self.output_dir)}"
        assert isinstance(self.num_templates,
                          int), f"num_templates must be an integer, found: {type(self.num_templates)}"
        assert isinstance(self.num_recycles, int), f"num_recycles must be an integer, found: {type(self.num_recycles)}"
        assert isinstance(self.models, list) and all(isinstance(m, str) for m in self.models), (
            f"models must be a list of strings, found: {self.models}"
        )
        required_columns = {'peptide', 'mhc_seq', 'mhc_type', 'anchors', 'id'}
        missing_columns = required_columns - set(self.df.columns)
        assert not missing_columns, f"df is missing required columns: {missing_columns}"
        assert self.df['peptide'].apply(lambda x: isinstance(x, str)).all(), "All peptide values must be strings."
        assert self.df['mhc_seq'].apply(lambda x: isinstance(x, str)).all(), "All mhc_seq values must be strings."
        assert self.df['mhc_type'].apply(
            lambda x: x in [1, 2]).all(), "MHC type must be either 1 (MHC-I) or 2 (MHC-II)."
        def valid_anchor_format(anchor, mhc_type):
            if pd.isna(anchor):
                return True  # Allow missing anchors (to be predicted)
            parts = anchor.split(";")
            return (mhc_type == 1 and len(parts) == 2) or (mhc_type == 2 and len(parts) == 4)
        assert self.df.apply(lambda row: valid_anchor_format(row['anchors'], row['mhc_type']), axis=1).all(), (
            "Anchors must be two numbers separated by ';' for MHC-I and four for MHC-II."
        )
        assert isinstance(self.alphafold_param_folder,
                          str), f"alphafold_param_folder must be a string, found: {type(self.alphafold_param_folder)}"
        assert os.path.isdir(
            self.alphafold_param_folder), f"alphafold_param_folder does not exist or is not a directory: {self.alphafold_param_folder}"
        assert isinstance(self.fine_tuned_model_path,
                          str), f"fine_tuned_model_path must be a string, found: {type(self.fine_tuned_model_path)}"
        assert not any('_ft' in name for name in self.models) or os.path.isfile(
            self.fine_tuned_model_path), f"fine_tuned_model_path does not exist or is not a file: {self.fine_tuned_model_path}"
