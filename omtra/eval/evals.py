from omtra.eval.register import register_inference_eval, register_train_eval
import peppr


@register_train_eval("exp_apo_conditioned_denovo_ligand")
def clash():
    return peppr.ClashCount()


@register_train_eval("exp_apo_conditioned_denovo_ligand")
def bond_length_violations():
    return peppr.BondLengthViolations()


@register_train_eval("exp_apo_conditioned_denovo_ligand")
def pocket_aligned_ligand_rmsd():
    return peppr.PocketAlignedLigandRMSD()


@register_train_eval("exp_apo_conditioned_denovo_ligand")
def ligand_rmsd():
    return peppr.LigandRMSD()
