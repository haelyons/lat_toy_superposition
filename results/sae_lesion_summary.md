# SAE-feature lesion on a fixed LM — summary

Model distilgpt2, layer 4, SAE dict 3072/k32. 160 features ablated. Paradigm-native: latent ablation on a fixed model, no retraining. Does edit collateral track reliance and interference?

- r(collateral, reliance_freq) = +0.52
- r(collateral, interference) = +0.02
- r(collateral, reliance) genuine-knockouts = +0.61
- mean collateral = 0.0202, mean self_effect = 0.1457
- collateral / self_effect ratio = 0.138 (low = edits are surgical)

Read: a positive reliance/interference correlation = the same axes that governed edit-cleanliness in the toy + transformer also govern it for real SAE features — i.e. steering/ablation cleanliness is a measurable property of the representation.