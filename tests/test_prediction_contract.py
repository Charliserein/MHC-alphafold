"""输入与恢复协议测试，不要求 GPU。"""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mhcalphafold.inputs import anchor_combinations, expand, normalize, read_inputs
from mhcalphafold.outputs import load_checkpoint, save_best, trim_scores, validate_candidate


def sample(**changes):
    return dict(dict(id='sample_1', peptide='ACDEFGHI', mhc_seq='A' * 80, mhc_type=1, anchors='2;8'), **changes)


class InputContract(unittest.TestCase):
    def test_mhc1_enumeration_covers_three_eightmer_choices(self):
        rows = expand([normalize(sample(anchors=''))])
        self.assertEqual([r['anchors'] for r in rows], ['1;7', '1;8', '2;8'])
        self.assertEqual({r['source_id'] for r in rows}, {'sample_1'})

    def test_given_anchor_is_preserved(self):
        rows = expand([normalize(sample())])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['anchors'], '2;8')

    def test_mhc2_requires_two_chains_and_four_anchors(self):
        row = normalize(sample(mhc_type=2, mhc_seq='A' * 80 + '/' + 'C' * 90,
                               peptide='ACDEFGHIKL', anchors='1;4;6;9'))
        self.assertEqual(row['mhc_seq'].count('/'), 1)
        self.assertEqual(anchor_combinations(10, 2), [(1, 4, 6, 9), (2, 5, 7, 10)])
        with self.assertRaises(ValueError):
            normalize(sample(mhc_type=2))

    def test_rejects_path_escape_ambiguous_sequence_and_bad_anchors(self):
        for changes in [dict(id='../escape'), dict(peptide='AXDEFGHI'), dict(anchors='8;2'),
                        dict(anchors='2;9'), dict(mhc_seq='A' * 34), dict(anchors='2;2')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalize(sample(**changes))

    def test_duplicate_ids_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(sample()))
                writer.writeheader()
                writer.writerows([sample(), sample()])
            args = argparse.Namespace(input=str(path), peptide=None, mhc_seq=None, mhc_fasta=None, anchors=None)
            with self.assertRaisesRegex(ValueError, '重复'):
                read_inputs(args)


class OutputContract(unittest.TestCase):
    def test_padding_does_not_contaminate_real_residue_scores(self):
        plddt, pae = trim_scores(np.array([90., 80., np.nan]),
                                np.array([[0., 1., np.nan], [1., 0., np.nan], [np.nan] * 3]), 2)
        self.assertEqual(plddt.tolist(), [90., 80.])
        self.assertEqual(pae.shape, (2, 2))

    def test_truncated_or_nonfinite_real_scores_rejected(self):
        with self.assertRaises(ValueError):
            trim_scores(np.ones(2), np.ones((2, 2)), 3)
        with self.assertRaises(ValueError):
            trim_scores(np.array([90., np.nan]), np.ones((2, 2)), 2)

    def test_saved_outputs_are_trimmed_and_tampering_invalidates_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = normalize(sample(peptide='AAAAAAAA'))
            candidate = dict(row, id='sample_1__a0000', source_id=row['id'])
            pdb = root / 'sample_1__a0000_model_1_model_2_ptm.pdb'
            length = len(row['mhc_seq']) + len(row['peptide'])
            lines = [f'ATOM  {i:5d}  CA  ALA A{i:4d}    {float(i):8.3f}{0.:8.3f}{0.:8.3f}  1.00 90.00           C\n'
                     for i in range(1, length + 1)]
            pdb.write_text(''.join(lines) + 'END\n')
            prefix = str(pdb.with_suffix(''))
            np.save(prefix + '_plddt.npy', np.full(length + 4, 90.))
            np.save(prefix + '_predicted_aligned_error.npy', np.ones((length + 4, length + 4)))
            np.save(prefix + '_ptm.npy', np.array(.9))
            scored = validate_candidate(candidate, pdb)
            self.assertEqual(scored['model'], 'model_2_ptm')
            result = save_best(root, row, [scored], 'fingerprint')
            from Bio.PDB import PDBParser
            chains = list(PDBParser(QUIET=True).get_structure('export', root / result['pdb_path']).get_chains())
            self.assertEqual([(chain.id, len(list(chain.get_residues()))) for chain in chains], [('A', 80), ('B', 8)])
            self.assertEqual([a.coord.tolist() for a in chains[1].get_atoms()],
                             [[float(i), 0., 0.] for i in range(81, 89)])
            self.assertEqual(np.load(root / result['plddt_path']).shape, (length,))
            self.assertIsNotNone(load_checkpoint(root, row['id'], 'fingerprint'))
            self.assertIsNone(load_checkpoint(root, row['id'], 'different-config'))
            (root / result['pae_path']).write_bytes(b'corrupted')
            self.assertIsNone(load_checkpoint(root, row['id'], 'fingerprint'))


if __name__ == '__main__':
    unittest.main()
