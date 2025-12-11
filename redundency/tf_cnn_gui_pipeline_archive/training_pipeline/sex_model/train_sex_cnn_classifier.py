import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                     GlobalAveragePooling2D, Dense, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ================== 1. load data ==================
data_dir = r"/Users/yanyanru/desktop/genotype/train2/sex_binary_640x512_npy"

X_train = np.load(os.path.join(data_dir, "X_train.npy"))  # [Ntr, H,W,1], uint8
y_train = np.load(os.path.join(data_dir, "y_train.npy"))  # one-hot [Ntr, 2]
X_test  = np.load(os.path.join(data_dir, "X_test.npy"))   # [Nte, H,W,1]
y_test  = np.load(os.path.join(data_dir, "y_test.npy"))   # one-hot [Nte, 2]

print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

num_classes = y_train.shape[1]   
print("num_classes =", num_classes)

# ================== 2. preprocessesing ==================
# Convert to float32 and normalize to 0 to 1
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

input_shape = X_train.shape[1:]   # (H,W,1)
print("input_shape =", input_shape)

batch_size = 16

# ================== 3. tf.data Data pipeline + data enhancement ==================
data_augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.03),
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

train_ds = make_dataset(X_train, y_train, training=True)
test_ds  = make_dataset(X_test,  y_test,  training=False)   # Both val and test

# ================== 4. Simple CNN Gender Dichotomy Model ==================
inputs = Input(shape=input_shape)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
#  Can also add another layer of Conv2D(256,...)

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation='relu')(x)
# x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.4)(x)
outputs = Dense(num_classes, activation='softmax')(x)   # num_classes=2

model = Model(inputs, outputs, name="wing_sex_cnn")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
 
model.summary()

json_string =model.to_json()  
open('architecture.json','w').write(json_string)
 
# ================== 5. Training (val is test_ds) ==================
ckpt_path = os.path.join( "sex_cnn_weights11.h5")

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
        patience=1000,
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
model.load_weights('sex_cnn_weights2.h5')
history = model.fit(
    train_ds,
    epochs=2000,
    validation_data=test_ds,   # val = test
    callbacks=callbacks,
)
   
# ================== 6. Evaluate on the test set ==================
print("\n=== Evaluate on TEST (as the test set) ===")
test_loss, test_acc = model.evaluate(test_ds)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# ================== 7. Generate a classification report & confusion matrix ==================
from sklearn.metrics import classification_report, confusion_matrix

y_true_onehot = np.concatenate([y for _, y in test_ds], axis=0)
y_true = np.argmax(y_true_onehot, axis=1)

y_pred_prob = model.predict(test_ds)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n=== Classification report ===")
print(classification_report(y_true, y_pred, digits=4))

print("\n=== Confusion matrix ===")
print(confusion_matrix(y_true, y_pred))

# ================== 8. Plotting the training curve ==================
plt.figure(figsize=(10, 4))
 
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc (test)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss (test)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")
 
plt.tight_layout()
plt.show()
 
# save the weights
model.save_weights(os.path.join("sex_cnn_weights121.h5"))
