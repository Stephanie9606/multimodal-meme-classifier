# Main Results (reproduced from paper)

Training accuracy comparison across the four model families, binary vs
multiclass setting. Numbers are read directly from the figures in the
IEEE paper (10825758) and reflect **training accuracy** at epoch 1 and
epoch 10; the paper discusses convergence rather than final test
accuracy in detail.

Binary task = {"Who Killed Hannibal", "Scared Cat"}.
Multiclass task = {"Who Killed Hannibal", "Scared Cat", "Sleeping Shaq", "Uncle Sam", "Peter Parker Cry"}.

| Model                | Task       | Epoch-1 acc | Epoch-10 acc |
| -------------------- | ---------- | :---------: | :----------: |
| VGG-16 (image only)  | Binary     |    0.93     |     0.99     |
| VGG-16 (image only)  | Multiclass |    0.32     |     0.99     |
| BERT  (text only)    | Binary     |  ~0.80–0.85 |     0.99     |
| BERT  (text only)    | Multiclass |  ~0.80      |     0.99     |
| Early fusion         | Binary     |    ~0.50    |     0.99     |
| Early fusion         | Multiclass |    0.38     |     0.99     |
| Late fusion          | Binary     |    ~0.50    |     0.99     |
| Late fusion          | Multiclass |    0.22     |     0.99 (reached at epoch 8) |

**Key findings from the paper:**

* Early fusion converges faster than late fusion on the multiclass task
  (epoch-1 accuracy 0.38 vs 0.22) while reaching a comparable final
  training accuracy.
* Binary models outperform multiclass on a per-epoch basis because the
  decision boundary is simpler.
* The late-fusion gap is attributed to late fusion operating on
  inference scores rather than joint feature embeddings — it can miss
  lower-level text/image interactions.
