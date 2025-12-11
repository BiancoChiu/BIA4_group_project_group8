import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import resnet
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Dense,
                                     Dropout, Conv2D, MaxPooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ================== 1. load data ==================
data_dir = r"/Users/yanyanru/desktop/genotype/train2/dataset_640512_npy"

X_train = np.load(os.path.join(data_dir, "X_train.npy"))  # [Ntr, H,W,1], uint8
y_train = np.load(os.path.join(data_dir, "y_train.npy"))  # one-hot [Ntr, C]



X_test  = np.load(os.path.join(data_dir, "X_test.npy"))   # [Nte, H,W,1]
y_test  = np.load(os.path.join(data_dir, "y_test.npy"))   # one-hot [Nte, C]
 
print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

num_classes = y_train.shape[1]
print("num_classes =", num_classes)

# ================== 2. Preprocessing: gray-scale to 3-channel ==================
X_train = X_train.astype("float32")
X_test  = X_test.astype("float32")

# Copy to a 3-channel format, compatible with the (H,W,3) input of ResNet50
X_train_rgb = np.repeat(X_train, 3, axis=-1)
X_test_rgb  = np.repeat(X_test, 3, axis=-1)

orig_input_shape = (X_train_rgb.shape[1],
                    X_train_rgb.shape[2],
                    X_train_rgb.shape[3])   # Original dimensions, for example (512,640,3)

print("orig_input_shape =", orig_input_shape)

batch_size = 16

# ================== 3. tf.data Data pipeline + data enhancement ==================
data_augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.02),
        tf.keras.layers.RandomZoom(0.1),
    ],
    name="data_augment",
)

def make_dataset(X, y, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y)) 
    if training:
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.map(
            lambda x, label: (data_augment(x, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(X_train_rgb, y_train, training=True)
test_ds  = make_dataset(X_test_rgb,  y_test,  training=False)

# ================== 4. First, use CNN+Pooling to reduce the size by a factor of 4, and then pass the data to ResNet50 ==================
# Assuming the original dimensions are H and W, after applying two MaxPool2D operations with a kernel size of (2,2)，H/4, W/4
# When building ResNet50 here, enter the shape and set it by zooming out

# The size after 4x reduction is calculated based on the orig_input_shape first
h_small = orig_input_shape[0] // 2
w_small = orig_input_shape[1] // 2
resnet_input_shape = (h_small, w_small, 3)
print("ResNet50 input shape:", resnet_input_shape)

# ResNet50 Basic model (using ImageNet pre-trained weights)
base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=resnet_input_shape,
)
   
# Freeze ResNet parameters to prevent overfitting in the first place; Later can be thawed in part if needed
base_model.trainable = True

inputs = Input(shape=orig_input_shape)   # Original large image input

# ----- CNN + Pooling reduce 4 times-----
x = Conv2D(3, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)             # /2
x = Dropout(0.2)(x)
 
# now x.shape ≈ (H/4, W/4, 3)，is the same as resnet_input_shape 
x = resnet.preprocess_input(x)         

x = base_model(x, training=True)       # When freezing，training=False 
 

x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.6)(x)
x = Dense(128)(x)
x = Dropout(0.8)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs, name="wing_resnet50_smallinput")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
     
model.summary()
 
# ================== 5. Training (val is test_ds) ==================
ckpt_path = os.path.join( "geno_weights.h5")

callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ), 
    EarlyStopping(
        monitor="val_loss",
        patience=500,          
        restore_best_weights=True,
        verbose=1,
    ), 
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=100,
        min_lr=1e-6,
        verbose=1,
    ),
] 

json_string =model.to_json()  


open('geno_architecture1.json','w').write(json_string)
history = model.fit(
    train_ds,
    epochs=1000,
    validation_data=test_ds,
    callbacks=callbacks,
)


# model.load_weights('/Users/yanyanru/desktop/genotype/gui/architectureandweights/geno_weights.h5')
# ================== 6. Evaluate on the test set ==================
print("\n=== Evaluate on TEST (as test sets) ===")
test_loss, test_acc = model.evaluate(test_ds)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# ================== 7. Output classification report & confusion matrix ==================
y_true_onehot = np.concatenate([y for _, y in test_ds], axis=0)
y_true = np.argmax(y_true_onehot, axis=1)

y_pred_prob = model.predict(test_ds)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n=== Classification report ===")
print(classification_report(y_true, y_pred, digits=4))

print("\n=== Confusion matrix ===")
print(confusion_matrix(y_true, y_pred))

# ================== 8. Draw a training curve (including training loss) ==================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc (test)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train_loss")       # training loss
plt.plot(history.history["val_loss"], label="val_loss (test)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")

plt.tight_layout()
plt.show()


print("\n=== Training / Validation loss per epoch ===")
for i, (l, vl) in enumerate(zip(history.history["loss"],
                                history.history["val_loss"])):
    print(f"Epoch {i+1:03d}: loss={l:.4f}, val_loss={vl:.4f}")
# model.save_weights('1d2dweights4.h5')
