# Cross-Questions & Answers — Indian Sign Language Recognition

Viva preparation for the ISL recognition project. Answers are written the way you should
say them out loud: one direct sentence first, then the justification. Every number here
comes from `artifacts/` in this repository.

## A. Problem framing

**1. Why sign language recognition?**
India has roughly 63 million people with hearing loss and only a few hundred certified ISL
interpreters, so most deaf signers cannot communicate with a hearing person without a third
party. A camera-based translator removes that dependency. It is also a sequence problem, not
a single-image problem, which makes it technically more interesting than plain classification.

**2. Isn't this just gesture recognition?**
Gesture recognition usually means a handful of static poses. A sign is defined by five
simultaneous parameters — handshape, orientation, location, movement and non-manual markers
(facial expression) — and unfolds over time. Two signs can share a handshape and differ only
in movement, so the model must be temporal.

**3. Why Indian Sign Language, when ASL datasets are bigger?**
ISL is a distinct language with its own grammar, and is heavily two-handed where ASL is largely
one-handed, so an ASL model transfers poorly. ISL is also severely under-resourced.

**4. What is the scope — words or sentences?**
Isolated word-level recognition: one clip in, one word out. Continuous sentence recognition
needs segmentation of a signing stream plus a language model, which is the next phase
(CTC loss or a seq2seq decoder).

## B. Dataset

**5. Which dataset and why?**
The Indian Sign Language Video Dataset (Kaggle, MIT licence, 3.48 GB): 61 word signs,
3,630 clips, recorded as 640×720 webcam video at 30 fps, about 6.8 s per clip. It is video,
which the problem requires — an alphabet-image dataset would reduce this to static image
classification.

**6. How is it structured?**
One folder per word. Each word has 20 recorded takes, and every take also ships as a
left-tilt and a right-tilt variant, giving 60 clips per class. We extract landmarks from the
1210 original takes and keep the tilt variants as a separate pool.

**7. Is it balanced?**
Yes — exactly 20 takes per class by construction. The split is stratified per class, and we
report macro-F1 as well as accuracy so no class can hide.

**8. How do you prevent data leakage?**
Two mechanisms. The split is computed once with a fixed seed before any training, and the test
set is read only by the evaluation script. More importantly the split is *group-aware*: the
three tilt variants of one take are near-duplicates, so all of them go to the same side of the
split. Splitting them independently would test the model on clips it had effectively memorised.

**9. What is wrong with this dataset — and did you notice before or after training?**
After the first run, which is the honest answer. Every class was recorded by a single signer in
a single session lasting about five minutes — you can read it off the filenames
(`WIN_20231102_12_38_17` … `12_42_54`). The 20 takes of a word are therefore near-copies in the
same clothing, lighting and camera position. The benchmark is saturated: our model scores
100.0% and a Random Forest on time-pooled features also scores
100.0%.

**10. So is your 100% meaningless?**
It means the pipeline is correct end to end, and nothing more — it cannot rank models. That is
why we added three harder protocols: data efficiency, perturbation robustness, and an
augmentation ablation. Those are the numbers we actually defend.

**11. Why not do a signer-independent split?**
Because the dataset has one signer, so it is not possible. Each class also appears on exactly
one recording date, so date is perfectly confounded with class and a leave-session-out split is
impossible too. The right fix is a second dataset (INCLUDE, 4,292 videos over 263 signs from
multiple signers) for cross-dataset testing — that is our stated next step.

## C. Preprocessing & features

**12. Why keypoints instead of raw video?**
Three reasons. Data efficiency: a 3D-CNN on pixels needs hundreds of clips per class; we have
20 takes. Invariance: keypoints discard clothing, lighting, skin tone and background by
construction. Cost: a clip becomes a 64 × 478 matrix (~120 KB) instead of ~280 MB of pixels, so
training takes 2.2 minutes on a laptop GPU.

**13. What does MediaPipe give you?**
Per frame, a hand landmarker returns 21 3-D landmarks per hand plus handedness, and a pose
landmarker returns 33 body landmarks. We keep 9 upper-body pose points — nose, shoulders,
elbows, wrists, hips — and both hands.

**14. Describe your feature vector precisely.**
239 numbers per frame: 9 pose points as (x, y, visibility) = 27; both hands in body-normalised
(x, y) = 84; both hands in wrist-local, size-normalised (x, y, z) = 126; and 2 hand-present
flags. We append the frame-to-frame difference (velocity), giving 478 per frame, and resample
every clip to 64 frames. Final tensor: 64 × 478.

**15. What makes it signer-invariant?**
Every coordinate is translated to the shoulder mid-point and divided by shoulder width,
recomputed per frame. A tall signer far away and a short signer close up produce nearly
identical vectors. It is the single most important step in the project.

**16. Why both global and wrist-local hand coordinates?**
Location and handshape are separate linguistic parameters. Global coordinates say where the
hand is relative to the body; wrist-local coordinates say what shape it makes, independent of
where it is. Supplying both means the network does not have to disentangle them.

**17. Why velocity features — can't the LSTM learn them?**
It can, but it would spend capacity doing so. Differencing is free, removes the residual static
offset and speeds up convergence — the same reasoning as delta-MFCCs in speech.

**18. Why resample to 64 frames instead of padding and masking?**
Signs vary in duration but not in structure; a slow and a fast version of a sign are the same
sign time-warped. Resampling normalises tempo and gives fixed-size tensors, which also lets the
classical baselines consume the identical representation. Our robustness test confirms it
works: changing signing speed by ×0.5 / ×1.5 costs one test clip of 182 (98.4% vs 98.9% clean).

**19. What if a hand is not detected?**
Its 63 numbers are zeroed and the presence flag goes to 0, so the model can distinguish
"absent" from "at the origin". Short dropouts are additionally smoothed by resampling.

**20. Why keep only every third frame?**
Consecutive frames at 30 fps are nearly identical. At every third frame we still have ~68
frames per clip (10 fps effective), which is above the rate at which sign movement aliases, and
extraction over 1,210 clips drops from about 90 minutes to under 10.

## D. Model

**21. Why a BiLSTM with attention?**
The LSTM models temporal dependency; bidirectional because an offline clip is complete, so the
end of a sign may inform the beginning. Attention pooling learns which frames carry the sign —
every clip has idle lead-in and lead-out that mean-pooling would blend in.

**22. Why not a GRU, or a Transformer?**
A GRU is a reasonable alternative with fewer parameters and is a drop-in swap. A Transformer
encoder is implemented in `models.py` as an ablation; self-attention needs more data to beat a
recurrent model at this scale, where the LSTM's inductive bias helps.

**23. Explain the attention mathematically.**
A small MLP scores each frame's hidden state, scores are softmax-normalised over time, and the
clip vector is the weighted sum of hidden states. Additive (Bahdanau-style) attention with a
single query — not multi-head self-attention.

**24. How many parameters?**
1.67 M — small enough to train in minutes and run live, with regularisation sized for a small
dataset: dropout 0.3, label smoothing 0.1, weight decay 1e-4, gradient clipping, early stopping.

**25. Why keep SVM and Random Forest?**
As honest baselines on the same features, pooled over time with mean/std/min/max. SVM reaches
97.8% and Random Forest 100.0% on the standard split — which is exactly how we
discovered the benchmark was saturated. Reporting only the deep model would have hidden that.

**26. Why is pooling a weakness for the baselines?**
Mean/std/min/max are order-invariant: a sign and the same sign played backwards give identical
features. Any pair of classes differing only in movement direction is unrecoverable for them.

## E. Results — know these numbers

**27. What are your headline results?**
On the standard split, 99.6% ± 0.5 test top-1 over three seeds. Under reduced
supervision: 48.9% from one take per word, 73.6% from two, 95.6% from five, across
61 classes where chance is 1.6%.

**28. Why report a data-efficiency curve at all?**
Because the saturated benchmark cannot separate models, and because it answers the question a
deployer actually asks: how many recordings do I need to add a new word? Five takes per word
gets you to 95.6%.

**29. How much of the curve is seed noise?**
Repeating the full-data run over three seeds gives 99.6% ± 0.5 (98.9%, 100%, 100%), so
single-run differences below about a point carry no signal — k=10 and k=14 tie at 98.9%.
We say so rather than cherry-picking the best seed.

**30. What does the robustness test show?**
The trained model is re-tested on the same clips after perturbation. ±10% off-centring and
landmark noise leave accuracy at 98.9%, unchanged from clean; rotation 15° and ±20% scale cost
one clip of 182 (98.4%) — the body-frame normalisation doing its job. A mirrored signer costs
3.3 points (95.6%). Dropping 30% of frames costs little (96.7%). Occluding one hand collapses
it to 25.8%.

**31. Why is the occlusion number so bad? Isn't that a failure?**
No — it is the expected and correct behaviour. Many ISL signs are two-handed, so removing one
hand genuinely destroys the information; a model that still scored highly would be relying on
something other than the sign. It also tells us the model uses both hands rather than latching
onto one.

**32. Your augmentation made results worse. Explain.**
At two takes per class: 83.2% without augmentation, 75.5% with the full stack. Isolating the
cause, mirror-only is 74.7% and everything-except-mirror is 86.0%, so mirroring alone
accounts for the loss. The dataset has one signer with a consistent dominant hand, so mirrored
clips are inputs the test distribution never contains, and many ISL signs are not left-right
symmetric. Geometric jitter and noise do help: 86.5% versus 83.2%. We also tested the mirror
itself and found a bug: it swapped the hands but not the arms, so mirrored clips were
anatomically impossible. Fixed, it still costs 10.4 points (12.1 before), so the conclusion
never depended on the bug.

**33. Isn't a negative result a problem for your project?**
The opposite — it is measured, explained and actionable. The general lesson is that an
augmentation is only valid if it imitates variation that exists in deployment. If the system
were deployed for left-handed signers, mirroring would become correct, and we would need
left-handed data rather than synthetic mirrors.

**34. Where do the remaining errors fall?**
Analysed on the 2-take model (73.6%), because the full model makes too few errors to study.
Typical confusions: "knife" → "turtle", "cabbage" → "injury", "onion" → "brinjal". They concentrate on signs sharing a handshape and body location and
differing in a short movement or a facial marker the model never sees.

**35. Why report top-5 as well as top-1?**
In a translation aid, a short candidate list is still useful, and the top-1 / top-5 gap says
whether errors are near-misses in a confusable cluster or total failures. At one take per word
top-1 is 48.9% but top-5 is 79.7%.

**36. Why macro-F1?**
It averages per-class F1 without frequency weighting, so a word the model never predicts cannot
be hidden by common words.

## F. Deployment

**37. Does it run in real time?**
Yes — a rolling 64-frame buffer, re-predicting every 5 frames. MediaPipe landmarking is the
bottleneck; the classifier head is sub-millisecond.

**38. How do you avoid speaking nonsense during idle movement?**
A softmax confidence threshold of 0.6, and a word is spoken only when the prediction changes.
Proper onset/offset detection with a motion-energy gate is the planned upgrade.

**39. Could this run on a phone?**
Yes. MediaPipe was built for mobile and the classifier is a couple of megabytes; exporting to
TFLite or Core ML runs the whole pipeline on-device, which also keeps the video private.

**40. Why does the live demo use the same code as training?**
`features.py` is shared, so the normalisation applied at inference is byte-for-byte the one
applied in training. A separate inference path is a classic source of silent train/serve skew.

## G. Theory questions they like

**41. Explain LSTM gates.**
Forget gate decides what to drop from the cell state, input gate what to add, output gate what
to expose. The additive cell-state path carries gradients across many timesteps without
vanishing.

**42. Why bidirectional, and when can't you use it?**
It reads the clip forwards and backwards so each frame's representation sees the whole sign.
You cannot use it in a strictly streaming system that must emit before the sign ends; our demo
sidesteps this by classifying a completed sliding window.

**43. What is the vanishing gradient problem?**
Repeated multiplication by the same Jacobian in a recurrent chain shrinks gradients
exponentially, so early frames get almost no learning signal. Gating provides a near-identity
path that keeps them alive.

**44. Why softmax with cross-entropy?**
Softmax turns logits into a distribution; cross-entropy is the negative log-likelihood of the
true class under it. The composition has the clean, numerically stable gradient
(prediction − one-hot).

**45. What does label smoothing do?**
It stops the network driving logits to infinity on a small dataset — a regulariser that also
produces better-calibrated confidences, which matters because the live demo thresholds on them.

**46. Precision or recall — which matters here?**
For an assistive translator a confident wrong word is worse than no word, so precision at the
operating threshold matters more. That is why the demo refuses to speak below 0.6.

**47. Curse of dimensionality — is it a problem?**
The pooled baseline vector is 1,912-dimensional with a few hundred training samples.
Standardisation plus the SVM's margin and the Random Forest's feature subsampling make it
workable; the deep model avoids the issue by never pooling.

## H. Limitations & ethics

**48. State your limitations honestly.**
Isolated words only; vocabulary limited to 61 classes; no facial / non-manual markers; a
single-signer dataset, so no signer-independent evaluation is possible; and degradation under
heavy occlusion or very low light where MediaPipe itself fails.

**49. What would you do with three more months?**
Cross-dataset evaluation on INCLUDE, a face-landmark channel for non-manual markers,
continuous signing with CTC, and an on-device mobile build.

**50. Any ethical considerations?**
Video of a person's face and hands is personal data, so inference should stay on-device. And
the system must be presented as an aid, never a replacement for a human interpreter — in
medical, legal or educational settings, errors carry real cost.
