# Early Stroke Predictor — PWA

Installable web app that runs the clinical stroke-risk classifier from the thesis
*"Enhanced Early Prediction Framework for Stroke using Deep Learning Algorithms"*
(Durga Devi P, SRMIST) in the phone's browser. Works offline after the first load.

**Research demonstration — not a medical device and not a diagnosis.**

## What's inside

```
public/             the app (deploy this folder)
  i18n.js           Tamil and English strings
  index.html        4 screens: Check · Result · How it works · Research
                    Tamil / English toggle in each screen header
  app.js            form logic + the neural network forward pass
  model.js          trained weights (attention → ReLU → sigmoid)
  sw.js             offline cache
  manifest.webmanifest, icons/, fonts/
training/
  train.py          how model.js was trained (Python, numpy + autograd)
  model.json        same weights + test metrics
  reference_cases.json  known inputs/outputs used to check the JS matches Python
firebase.json       Firebase Hosting config
```

## Model

* Inputs: the 6 features SSC-AE kept — hypertension, heart disease, work type,
  average glucose, BMI, smoking status. Unknown BMI → mean imputation (28.9).
* Glucose and BMI are min-max scaled, each factor gets an embedding, an attention
  layer gives each factor a weight αᵢ (shown on the result screen), then a 16-unit
  ReLU layer and a sigmoid output.
* Trained on the public Kaggle *Stroke Prediction* dataset (5,110 patients,
  249 strokes). Held-out test (1,022 patients): ROC AUC 0.74, accuracy 83%,
  sensitivity 44%, specificity 85% at threshold 0.585.
* Risk bands: Low < 0.40 ≤ Moderate < 0.585 ≤ High. The result screen quotes the
  real stroke rate in each band from the dataset (1.9% / 6.0% / 13.3%).

To retrain: download `healthcare-dataset-stroke-data.csv` from Kaggle, save it as
`training/data.csv`, then `pip install numpy pandas scikit-learn autograd` and
`python train.py`. Copy the new `model.json` into `public/model.js` as
`window.MODEL = { ... };`.

## Try it locally

```
cd public
python3 -m http.server 8080
```
Open http://localhost:8080 (service worker needs http://localhost or https).

## Deploy to Firebase Hosting (from this repo, no file uploads)

Firebase project: `early-stroke-predictor-app` (already set in `.firebaserc`).

**First time**, in Google Cloud Shell (https://shell.cloud.google.com), signed in as the
account that owns the Firebase project:

```
cd ~ && rm -rf esp
git clone https://github.com/veerabathiran-gif/services.git esp
cd esp
firebase deploy --only hosting --project early-stroke-predictor-app
```

If `firebase` is not found: `npm install -g firebase-tools`, then deploy again.
If it asks you to sign in: `firebase login --no-localhost`, follow the link, paste the
code back, then deploy again.

**Every time after that** — two lines, nothing to upload:

```
cd ~/esp && git pull
firebase deploy --only hosting --project early-stroke-predictor-app
```

The app is live at https://early-stroke-predictor-app.web.app

## Make an Android APK (PWABuilder)

1. Go to https://www.pwabuilder.com and paste your `https://early-stroke-predictor-app.web.app` URL.
2. It should score the manifest + service worker as ready. Click **Package for stores → Android**.
3. Download the zip. It contains a signed test APK you can sideload, and an `.aab`
   plus signing key for the Play Store. Keep the signing key safe.

Note for Play Store: health apps need a clear disclaimer (already in the app) and
Google may ask for the Health apps declaration.

## Updating

After changing any file in `public/`, bump `CACHE` in `public/sw.js`
(`stroke-predictor-v1` → `v2`) so installed copies pick up the change, then deploy again.
