from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, LeakyReLU, concatenate
from tensorflow.keras.models import Model

frame_length = 1022
n_fft = 1022
frame_step = 300
n_sample = 8000
sample_rate = 8000
n_frame  = (n_sample - frame_length)//frame_step +1
n_freq = n_fft//2 + 1


print("n_frame",n_frame)
print("n_freq",n_freq)

def unet_3blocks(input_size=(n_frame, n_freq, 2), num_classes=2):
    inputs = Input(shape=input_size)

    # Encoder Block 1
    c1 = Conv2D(16, (3, 3), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = LeakyReLU(alpha=0.2)(c1)
    c1 = Conv2D(16, (3, 3), padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = LeakyReLU(alpha=0.2)(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    # Encoder Block 2
    c2 = Conv2D(32, (3, 3), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = LeakyReLU(alpha=0.2)(c2)
    c2 = Conv2D(32, (3, 3), padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = LeakyReLU(alpha=0.2)(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    # Encoder Block 3 (mới thêm)
    c3 = Conv2D(64, (3, 3), padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = LeakyReLU(alpha=0.2)(c3)
    c3 = Conv2D(64, (3, 3), padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = LeakyReLU(alpha=0.2)(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    # Bottleneck
    b = Conv2D(128, (3, 3), padding='same')(p3)
    b = BatchNormalization()(b)
    b = LeakyReLU(alpha=0.2)(b)
    b = Conv2D(128, (3, 3), padding='same')(b)
    b = BatchNormalization()(b)
    b = LeakyReLU(alpha=0.2)(b)

    # Decoder Block 1 (mới thêm)
    u1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(b)
    u1 = concatenate([u1, c3])
    u1 = BatchNormalization()(u1)
    u1 = LeakyReLU(alpha=0.2)(u1)
    u1 = Conv2D(64, (3, 3), padding='same')(u1)
    u1 = BatchNormalization()(u1)
    u1 = LeakyReLU(alpha=0.2)(u1)

    # Decoder Block 2 (sửa lại từ Decoder Block 2 cũ)
    u2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(u1)
    u2 = concatenate([u2, c2])
    u2 = BatchNormalization()(u2)
    u2 = LeakyReLU(alpha=0.2)(u2)
    u2 = Conv2D(32, (3, 3), padding='same')(u2)
    u2 = BatchNormalization()(u2)
    u2 = LeakyReLU(alpha=0.2)(u2)

    # Decoder Block 3 (sửa lại từ Decoder Block 3 cũ)
    u3 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(u2)
    u3 = concatenate([u3, c1])
    u3 = BatchNormalization()(u3)
    u3 = LeakyReLU(alpha=0.2)(u3)
    u3 = Conv2D(16, (3, 3), padding='same')(u3)
    u3 = BatchNormalization()(u3)
    u3 = LeakyReLU(alpha=0.2)(u3)

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='linear')(u3)

    model = Model(inputs=inputs, outputs=outputs)
    return model

model = unet_3blocks(input_size=( n_frame, n_freq, 2), num_classes=2)