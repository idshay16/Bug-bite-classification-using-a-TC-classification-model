# Changes Recap — Last 3 Commits

> Code-level diff comparison across `notebooks/Multiclass_Classification.ipynb` and `miscellaneous_code/augment_data.py`.

---

## Commit 1 · `fe25bdc` — Multiclass notebook refactor

**Date:** Sat 9 May 2026  
**File:** `notebooks/Multiclass_Classification.ipynb`

### 1a. Imports — backbone & preprocessing imports

**Before:**
```python
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
from tensorflow.keras.applications.convnext import ConvNeXtXLarge
```

**After:**
```python
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M
from tensorflow.keras.applications.convnext import ConvNeXtBase
```

### 1b. Data pipeline — normalization & batch size (cell 23)

**Before:**
```python
train_datagen = ImageDataGenerator(rescale = 1.0/255., rotation_range = 10)
train_generator = train_datagen.flow_from_directory(train_folders, batch_size = 8, class_mode = 'categorical', target_size = (310, 310))

validation_datagen = ImageDataGenerator(rescale = 1.0/255.)
validation_generator = validation_datagen.flow_from_directory(val_folders, batch_size = 8, class_mode = 'categorical', target_size = (310, 310), shuffle = False)
```

**After:**
```python
train_datagen = ImageDataGenerator(rotation_range=10)
train_generator = train_datagen.flow_from_directory(train_folders, batch_size=8, class_mode='categorical', target_size=(310, 310))

validation_datagen = ImageDataGenerator()
validation_generator = validation_datagen.flow_from_directory(val_folders, batch_size=8, class_mode='categorical', target_size=(310, 310), shuffle=False)
```

### 1c. Custom preprocessing layers added (cell 28 — new)

**After:**
```python
@tf.keras.utils.register_keras_serializable()
class ConvNeXtPreprocessing(tf.keras.layers.Layer):
    """Applies ConvNeXt ImageNet normalization: scales [0,255] to ~[-2,2] using ImageNet mean/std."""
    def call(self, x):
        x = tf.cast(x, tf.float32) / 255.0
        mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        std  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        return (x - mean) / std

@tf.keras.utils.register_keras_serializable()
class ResNetV2Preprocessing(tf.keras.layers.Layer):
    """Applies ResNetV2 preprocessing: scales [0,255] to [-1,1]."""
    def call(self, x):
        return tf.cast(x, tf.float32) / 127.5 - 1.0
```

### 1d. ConvNeXt model definition — XLarge → Base (cell 31)

**Before:**
```python
tf.keras.backend.clear_session()
```

**After:**
```python
convnext_backbone = ConvNeXtBase(weights='imagenet', include_top=False, pooling='avg')
convnext_backbone.trainable = False

inputs = keras.Input(shape=(310, 310, 3))
x = ConvNeXtPreprocessing()(inputs)
x = convnext_backbone(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(5, activation='softmax')(x)
convnext_final_model = Model(inputs, x)
```

### 1e. ResNet training — added compile() and ModelCheckpoint (cell 44)

**After:**
```python
# Phase 1: freeze backbone, train new top layers
resnet152v2_final_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['acc'])
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    f'{PROJECT_ROOT}/Model_Weights/multiclass_resnet152v2_best.keras',
    monitor='val_acc', save_best_only=True, verbose=1
)

resnet152v2_backbone.trainable = False
history = resnet152v2_final_model.fit(
    train_generator, epochs=resnet_phase1_epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr, checkpoint], verbose=1
)

# Phase 2: unfreeze top 60 layers for fine-tuning
resnet152v2_backbone.trainable = True
for layer in resnet152v2_backbone.layers[:-60]:
    layer.trainable = False
resnet152v2_final_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00001), metrics=['acc'])
history = resnet152v2_final_model.fit(
    train_generator, epochs=resnet_phase2_epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr, checkpoint], verbose=1
)
```

### 1f. EfficientNet model — V2L → V2M with include_preprocessing (cell 52)

**Before:**
```python
efficientnetv2l_backbone = EfficientNetV2L(input_shape = (310, 310, 3), weights = 'imagenet', include_top = False)
for layer in efficientnetv2l_backbone.layers:
    layer.trainable = True
efficientnetv2l_last_output = efficientnetv2l_backbone.output
efficientnetv2l_maxpooled_output = Flatten()(efficientnetv2l_last_output)
efficientnetv2l_x = Dense(1024, activation = 'relu')(efficientnetv2l_maxpooled_output)
efficientnetv2l_x = Dropout(0.5)(efficientnetv2l_x)
efficientnetv2l_x = Dense(5, activation = 'softmax')(efficientnetv2l_x)
efficientnetv2l_x_final_model = Model(inputs = efficientnetv2l_backbone.input, outputs = efficientnetv2l_x)
```

**After:**
```python
efficientnetv2m_backbone = EfficientNetV2M(weights='imagenet', include_top=False, pooling='avg', include_preprocessing=True)
efficientnetv2m_backbone.trainable = False

inputs = keras.Input(shape=(310, 310, 3))
x = efficientnetv2m_backbone(inputs)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(5, activation='softmax')(x)
efficientnetv2m_final_model = Model(inputs, x)
```

### 1g. Feature map — model loading fix (cell 62)

**Before:**
```python
tf.keras.backend.clear_session()

convnext_best_model = load_model(f'{PROJECT_ROOT}/Model_Weights/multiclass_convnextxlarge_best.keras', compile=False)
resnet_best_model = load_model(f'{PROJECT_ROOT}/Model_Weights/multiclass_resnet152v2_model.keras', compile=False)
efficientnet_best_model = load_model(f'{PROJECT_ROOT}/Model_Weights/multiclass_efficientnetv2l_model.keras', compile=False)
```

**After:**
```python
from tensorflow.keras.models import load_model as _load_model
tf.keras.backend.clear_session()
model = _load_model(f'{PROJECT_ROOT}/Model_Weights/multiclass_resnet152v2_model.keras', compile=False)
```

### 1h. Feature map — extractor rewired through full pipeline (cell 66)

**After:**
```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from numpy import expand_dims

# Find preprocessing layer and backbone sub-model (nested Functional)
preproc  = next((l for l in model.layers if 'preprocessing' in l.name.lower()), None)
backbone = next(l for l in model.layers if hasattr(l, 'layers') and len(getattr(l, 'layers', [])) > 5)

# Find first conv layer with ≥32 channels inside the backbone
target_layer = None
for layer in backbone.layers:
    if 'conv' in layer.name and len(layer.output.shape) == 4 and layer.output.shape[-1] >= 32:
        target_layer = layer
        break
if target_layer is None:
    for layer in backbone.layers:
        if 'conv' in layer.name and len(layer.output.shape) == 4:
            target_layer = layer
            break

print(f"Visualising layer: {target_layer.name}, shape: {target_layer.output.shape}")

# Build a sub-model from backbone.inputs → target conv layer
backbone_feat = Model(inputs=backbone.inputs, outputs=target_layer.output)

# Re-wire through model.input so the full preprocessing chain is included:
#   model.input → preproc layer → backbone_feat → feature maps
preproc_out = preproc(model.input)
feat_out    = backbone_feat(preproc_out)
feat_model  = Model(inputs=model.input, outputs=feat_out)

# Load image normalized to [0,1] — same as ImageDataGenerator rescale=1/255 during training
img = load_img(img_path, target_size=(310, 310))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = img / 255.0
feature_maps = feat_model.predict(img)

n_maps = feature_maps.shape[3]
n_display = min(32, n_maps)
cols = 4
rows = (n_display + cols - 1) // cols

ix = 1
plt.figure(figsize=(10, rows * 2.5))
for _ in range(rows):
    for _ in range(cols):
        if ix > n_display:
            break
        ax = plt.subplot(rows, cols, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(feature_maps[0, :, :, ix-1], cmap='viridis')
        ix += 1
plt.tight_layout()
plt.savefig("Feature Maps.jpg", transparent=True, bbox_inches='tight')
plt.show()
```

### 1i. Ensemble prediction loop — restored with tiebreaker fix (cell 69)

**Before:**
```python
def mode(my_list):
    ct = Counter(my_list)
    max_value = max(ct.values())
    return ([key for key, value in ct.items() if value == max_value])

true_value = []
combined_model_pred = []
convnext_pred = []
resnet_pred = []
efficientnet_pred = []

for folder in os.listdir(val_folders):
    test_image_ids = [f for f in os.listdir(os.path.join(val_folders,folder)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for image_id in test_image_ids[:int(len(test_image_ids))]:
        path = os.path.join(val_folders, folder, image_id)
        img_raw = cv2.imread(path)
        if img_raw is None:
            continue

        true_value.append(validation_generator.class_indices[folder])
        img = cv2.resize(img_raw, (310, 310))
        img_normalized = img / 255

        convnext_image_prediction = np.argmax(convnext_best_model.predict(np.array([img_normalized])))
        convnext_pred.append(convnext_image_prediction)

        resnet_image_prediction = np.argmax(resnet_best_model.predict(np.array([img_normalized])))
        resnet_pred.append(resnet_image_prediction)

        efficientnet_image_prediction = np.argmax(efficientnet_best_model.predict(np.array([img_normalized])))
        efficientnet_pred.append(efficientnet_image_prediction)

        mode_result = mode([convnext_image_prediction, resnet_image_prediction, efficientnet_image_prediction])
        image_prediction = resnet_image_prediction if len(mode_result) > 1 else mode_result[0]
        combined_model_pred.append(image_prediction)
```

**After:**
```python
from collections import Counter

def mode(my_list):
    ct = Counter(my_list)
    max_value = max(ct.values())
    return [key for key, value in ct.items() if value == max_value]

true_value = []
combined_model_pred = []
convnext_pred = []
resnet_pred = []
efficientnetv2m_pred = []

for folder in os.listdir(val_folders):
    folder_path = os.path.join(val_folders, folder)
    if not os.path.isdir(folder_path) or folder not in validation_generator.class_indices:
        continue
    test_image_ids = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for image_id in test_image_ids:
        path = os.path.join(folder_path, image_id)
        img_raw = cv2.imread(path)
        if img_raw is None:
            continue

        true_value.append(validation_generator.class_indices[folder])
        img = cv2.resize(img_raw, (310, 310))
        img_normalized = img / 255.0

        convnext_pred_val = np.argmax(convnext_best_model.predict(np.array([img_normalized])))
        convnext_pred.append(convnext_pred_val)

        resnet_pred_val = np.argmax(resnet_best_model.predict(np.array([img_normalized])))
        resnet_pred.append(resnet_pred_val)

        efficientnetv2m_pred_val = np.argmax(efficientnetv2m_best_model.predict(np.array([img_normalized])))
        efficientnetv2m_pred.append(efficientnetv2m_pred_val)

        mode_result = mode([convnext_pred_val, resnet_pred_val, efficientnetv2m_pred_val])
        image_prediction = resnet_pred_val if len(mode_result) > 1 else mode_result[0]
        combined_model_pred.append(image_prediction)
```

---

## Commit 2 · `d133f32` — Output formatting & augment_data refactor

**Date:** Fri 8 May 2026  
**Files:** `miscellaneous_code/augment_data.py`, `training_file_yolov3.ipynb` (output metadata only)


> `training_file_yolov3.ipynb` changes were metadata-only (execution counts, stream output type) — no code logic changed.


### augment_data.py — diff

```diff
@@ -13,6 +13,7 @@ import sys
 import random
 import argparse
 import math
+import shutil
 from pathlib import Path
 
 # Ensure UTF-8 output on Windows terminals
@@ -33,6 +34,14 @@ from collections import defaultdict
 SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
 
+def _longpath(p: Path) -> str:
+    """Prepend the Windows extended-length prefix to bypass MAX_PATH (260 chars)."""
+    if sys.platform != "win32":
+        return str(p)
+    resolved = str(p.resolve())
+    return resolved if resolved.startswith("\\\\") else "\\\\?\\" + resolved
+
+
 def find_splits(data_dir: Path) -> dict[str, Path]:
     """
     Return a dict of {split_name: Path} for any train/val/test sub-folders.
@@ -267,16 +276,15 @@ def balance_dataset(
     max_count   = target_count or max(counts.values())
     total_new   = sum(max(0, max_count - n) for n in counts.values())
 
-    # Destination root
-    if output_dir is None:
-        dest_root = data_dir / f"{split}_augmented"
-    else:
-        dest_root = output_dir
+    in_place = output_dir is None
 
     print("\n-- Balancing Plan ------------------------------------------")
     print(f"  Target count per class : {max_count}")
     print(f"  Total new images       : {total_new}")
-    print(f"  Output directory       : {dest_root}")
+    if in_place:
+        print(f"  Output                 : in-place (same class folders)")
+    else:
+        print(f"  Output directory       : {output_dir}")
     print()
 
     for cls_name, imgs in sorted(classes.items()):
@@ -296,42 +304,111 @@ def balance_dataset(
         print("Aborted.")
         return
 
-    dest_root.mkdir(parents=True, exist_ok=True)
+    if not in_place:
+        output_dir.mkdir(parents=True, exist_ok=True)
+
     total_written = 0
 
     for cls_name, src_imgs in sorted(classes.items()):
-        need     = max(0, max_count - len(src_imgs))
-        cls_dest = dest_root / cls_name
-        cls_dest.mkdir(parents=True, exist_ok=True)
-
-        # Copy originals
-        for src in src_imgs:
-            dst = cls_dest / src.name
-            if not dst.exists():
-                Image.open(src).save(dst)
+        need = max(0, max_count - len(src_imgs))
+
+        if in_place:
+            cls_dest = split_path / cls_name
+        else:
+            cls_dest = output_dir / cls_name
+            cls_dest.mkdir(parents=True, exist_ok=True)
+            for src in src_imgs:
+                dst = cls_dest / src.name
+                if not dst.exists():
+                    shutil.copy2(_longpath(src), _longpath(dst))
 
         if need == 0:
             print(f"  {cls_name:<20s}  [ok] no augmentation needed")
             continue
 
+        # Only augment originals, never re-augment already-augmented files
+        pool = [p for p in src_imgs if not p.name.startswith("augmented_")]
+        if not pool:
+            print(f"  {cls_name:<20s}  [skip] no original images found")
+            continue
+
+        # Start counter after any existing augmented files to avoid name collisions on re-runs
+        aug_counter = len(src_imgs) - len(pool)
+
         written = 0
-        pool    = list(src_imgs)
-        idx     = 0
+        src_idx = 0
         while written < need:
-            src_path = pool[idx % len(pool)]
-            idx += 1
+            src_path = pool[src_idx % len(pool)]
+            src_idx += 1
+            out_name = f"augmented_{aug_counter:04d}_{src_path.name}"
+            aug_counter += 1
             img = Image.open(src_path).convert("RGB")
             aug, _ = augment_image(img)
-            stem    = src_path.stem
-            out_name = f"{stem}_aug{written:04d}{src_path.suffix}"
-            aug.save(cls_dest / out_name)
+            aug.save(_longpath(cls_dest / out_name))
             written += 1
 
         total_written += written
         print(f"  {cls_name:<20s}  [ok] wrote {written} augmented images")
 
     print(f"\nDone. Total new images written: {total_written}")
-    print(f"Output: {dest_root}")
+    if in_place:
+        print(f"Output: in-place within {split_path}")
+    else:
+        print(f"Output: {output_dir}")
+
+
+# ──────────────────────────────────────────────────────────────────────────────
+# Option 4 – Remove augmented data
+# ──────────────────────────────────────────────────────────────────────────────
+
+def remove_augmented_data(
+    data_dir: Path,
+    output_dir: Path | None = None,
+    split: str = "train",
+    dry_run: bool = False,
+) -> None:
+    if output_dir is not None:
+        # Dedicated output dir was used — offer to delete the whole directory
+        if not output_dir.exists():
+            print(f"\nNothing to remove — directory not found: {output_dir}")
+            return
+        aug_files = [p for p in output_dir.rglob("*") if p.is_file()]
+        print("\n-- Cleanup Plan --------------------------------------------")
+        print(f"  Directory : {output_dir}")
+        print(f"  Files     : {len(aug_files)}")
+        if dry_run:
+            print("\n[dry-run] No files removed.")
+            return
+        confirm = input(f"\nDelete entire directory '{output_dir}'? [y/N] ").strip().lower()
+        if confirm != "y":
+            print("Aborted.")
+            return
+        shutil.rmtree(output_dir)
+        print(f"\nDone. Removed directory: {output_dir}")
+    else:
+        # In-place mode — find augmented_* files within the split's class folders
+        splits = find_splits(data_dir)
+        if split not in splits:
+            available = list(splits.keys())
+            split = available[0]
+        split_path = splits[split]
+        aug_files = [p for p in split_path.rglob("augmented_*") if p.is_file()]
+        print("\n-- Cleanup Plan --------------------------------------------")
+        print(f"  Split directory: {split_path}")
+        print(f"  Augmented files: {len(aug_files)}")
+        if dry_run:
+            print("\n[dry-run] No files removed.")
+     
```

---

## Commit 3 · `00b69a5` — Add data augmentation utility

**Date:** Fri 8 May 2026  
**File:** `miscellaneous_code/augment_data.py` *(new file)*
