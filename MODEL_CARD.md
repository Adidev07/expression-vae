# Model / system card

## Intended use

Train a variational autoencoder on low-dimensional fictional expression profiles and sample new mathematical profiles. Educational local experiment; no production deployment is implied.

## Data

Synthetic arrays, manually written documents or replayed JSON. No real user records. Seeds and sample inputs are recorded in app.py and sample_input.json.

## Method

Encode standardized expression through tanh into Gaussian latent means and log variances. Use reparameterization z=mu+exp(logvar/2)*epsilon and a linear decoder. Optimize squared reconstruction plus beta-weighted KL divergence with analytic gradients.

## Evaluation protocol

Train on 500 fictional profiles and evaluate reconstruction on 200 independent profiles. Compare with a training-mean baseline; save generated profiles and run a fixed-noise numerical gradient check.

## Known limits

Generated values are standardized mathematical profiles, not validated cell states, count measurements or treatment predictions. The model is not scVI and has no biological foundation-model pretraining.

## Verification status

Local tests and default experiment are executed during packaging. Live model generation is not exercised in the build. A successful validation result is limited to the checks implemented in code.

## Further work

Use appropriate count likelihoods, donor and batch covariates, posterior predictive checks and licensed expression data before biological interpretation.
