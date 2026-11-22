import contextlib
import io
import unittest
import numpy as np
from omtra.data.condensed_atom_typing import CondensedAtomTyper


class SharedVocabularyTest(unittest.TestCase):
    def test_crossdocked_preserves_token_ids_and_special_tokens(self):
        for fake in (False, True):
            plain = CondensedAtomTyper(fake_atoms=fake)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                crossdocked = CondensedAtomTyper(fake_atoms=fake, include_crossdocked=True)
            self.assertEqual(output.getvalue(), '')
            np.testing.assert_array_equal(plain.cond_to_uncond, crossdocked.cond_to_uncond)
            self.assertEqual(plain.fake_atom_idx, crossdocked.fake_atom_idx)
            self.assertEqual(plain.masked_atom_idx, crossdocked.masked_atom_idx)
            rows = plain.cond_to_uncond[:plain.n_real_categories]
            np.testing.assert_array_equal(
                crossdocked.feats_to_cond_a(rows[:, 0], rows[:, 1], rows[:, 2:]),
                np.arange(plain.n_real_categories))

    def test_unknown_tuple_fails_without_changing_vocabulary(self):
        typer = CondensedAtomTyper(fake_atoms=False, include_crossdocked=True)
        before = typer.cond_to_uncond.copy()
        row = before[:1].copy()
        row[0, 0] = -999
        with self.assertRaisesRegex(ValueError, 'shared Pharmit/PLINDER checkpoint vocabulary'):
            typer.feats_to_cond_a(row[:, 0], row[:, 1], row[:, 2:])
        np.testing.assert_array_equal(before, typer.cond_to_uncond)


if __name__ == '__main__':
    unittest.main()
