import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

IMG_SIZE = 224
IMAGE_CHANNELS = 3

CLINICAL_DIM = 8
CLINICAL_EMBED_DIM = 128

IMAGE_EMBED_DIM = 512
FUSION_DIM = 512

NUM_CLASSES = 2


# ============================================================
# 1. SPATIAL FEATURE EXTRACTION USING RESNET50
# ============================================================

def build_resnet50_backbone(
    input_shape=(IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS),
    trainable=False
):
    """
    ResNet50-based spatial feature extractor.

    Output:
        Spatial feature map from the final convolutional stage.
    """

    image_input = layers.Input(
        shape=input_shape,
        name="mri_input"
    )

    backbone = ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input
    )

    backbone.trainable = trainable

    # Keep the spatial feature map for
    # subsequent S-Transformer processing.
    feature_map = backbone.output

    return Model(
        image_input,
        feature_map,
        name="ResNet50_Spatial_Extractor"
    )


# ============================================================
# 2. S-TRANSFORMER
# ============================================================

class STransformer(layers.Layer):
    """
    Spatial Transformer block.

    The spatial feature map is converted into tokens and
    contextual relationships are learned using multi-head
    self-attention.
    """

    def __init__(
        self,
        embed_dim=2048,
        num_heads=8,
        ff_dim=4096,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.ffn = tf.keras.Sequential([
            layers.Dense(
                ff_dim,
                activation=tf.nn.gelu
            ),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ])

        self.dropout = layers.Dropout(
            dropout
        )

    def call(self, x, training=False):

        # x = [B, H, W, C]
        batch = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]

        # Flatten spatial positions into tokens
        tokens = tf.reshape(
            x,
            [batch, height * width, self.embed_dim]
        )

        # Self-attention
        attention_output = self.attention(
            tokens,
            tokens,
            training=training
        )

        tokens = self.norm1(
            tokens + self.dropout(
                attention_output,
                training=training
            )
        )

        # Feed-forward network
        ffn_output = self.ffn(
            tokens,
            training=training
        )

        tokens = self.norm2(
            tokens + self.dropout(
                ffn_output,
                training=training
            )
        )

        # Restore spatial structure
        output = tf.reshape(
            tokens,
            [batch, height, width, self.embed_dim]
        )

        return output


# ============================================================
# 3. CHANNEL-WISE REFINEMENT
# ============================================================

class ChannelWiseRefinement(layers.Layer):
    """
    Lightweight channel attention/refinement.

    Learns channel importance using global average pooling.
    """

    def __init__(
        self,
        reduction=8,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.reduction = reduction

    def build(self, input_shape):

        channels = int(input_shape[-1])

        hidden_dim = max(
            channels // self.reduction,
            1
        )

        self.fc1 = layers.Dense(
            hidden_dim,
            activation="relu"
        )

        self.fc2 = layers.Dense(
            channels,
            activation="sigmoid"
        )

    def call(self, x):

        pooled = tf.reduce_mean(
            x,
            axis=[1, 2]
        )

        weights = self.fc1(
            pooled
        )

        weights = self.fc2(
            weights
        )

        weights = tf.expand_dims(
            weights,
            axis=1
        )

        weights = tf.expand_dims(
            weights,
            axis=1
        )

        return x * weights


# ============================================================
# 4. GLOBAL FEATURE EMBEDDING
# ============================================================

class GlobalFeatureEmbedding(layers.Layer):
    """
    Converts the refined spatial representation into a
    compact 512-dimensional imaging embedding.
    """

    def __init__(
        self,
        embedding_dim=IMAGE_EMBED_DIM,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.pool = layers.GlobalAveragePooling2D()

        self.projection = layers.Dense(
            embedding_dim,
            activation="relu"
        )

        self.dropout = layers.Dropout(
            0.25
        )

    def call(self, x, training=False):

        x = self.pool(x)

        x = self.projection(x)

        x = self.dropout(
            x,
            training=training
        )

        return x


# ============================================================
# 5. CLINICAL VARIABLE ENCODING
# ============================================================

def encode_clinical_variables(
    clinical_data
):
    """
    Convert structured clinical variables into numeric
    representations.

    Expected columns:
        0 : Age
        1 : BI-RADS
        2 : Tumor Size
        3 : Breast Density
        4 : Menopausal Status
        5 : Family History
        6 : Previous Breast Cancer
        7 : Lesion Location

    The input should already follow the encoding scheme
    defined by the study dataset.
    """

    clinical_data = np.asarray(
        clinical_data,
        dtype=np.float32
    )

    if clinical_data.shape[-1] != CLINICAL_DIM:
        raise ValueError(
            f"Expected {CLINICAL_DIM} clinical features, "
            f"received {clinical_data.shape[-1]}"
        )

    return clinical_data


# ============================================================
# 6. CFLANN-BASED FUNCTIONAL EXPANSION
# ============================================================

class CFLANNFunctionalExpansion(
    layers.Layer
):
    """
    Functional expansion layer for the clinical branch.

    The layer maps the encoded clinical variables into a
    higher-dimensional nonlinear representation.
    """

    def __init__(
        self,
        expansion_dim=256,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.expansion_dim = expansion_dim

        self.linear_projection = layers.Dense(
            expansion_dim
        )

        self.nonlinear_projection = layers.Dense(
            expansion_dim,
            activation="tanh"
        )

        self.gate = layers.Dense(
            expansion_dim,
            activation="sigmoid"
        )

    def call(self, x):

        linear_features = (
            self.linear_projection(x)
        )

        nonlinear_features = (
            self.nonlinear_projection(x)
        )

        gate = self.gate(x)

        expanded = (
            gate * nonlinear_features
            +
            (1.0 - gate) * linear_features
        )

        return expanded


# ============================================================
# 7. CLINICAL FEATURE EMBEDDING
# ============================================================

class ClinicalFeatureEmbedding(
    layers.Layer
):
    """
    Generates compact clinical embedding.
    """

    def __init__(
        self,
        embedding_dim=CLINICAL_EMBED_DIM,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.expansion = (
            CFLANNFunctionalExpansion(
                expansion_dim=256
            )
        )

        self.embedding = layers.Dense(
            embedding_dim,
            activation="relu"
        )

        self.normalization = (
            layers.LayerNormalization()
        )

        self.dropout = layers.Dropout(
            0.20
        )

    def call(self, x, training=False):

        x = self.expansion(x)

        x = self.embedding(x)

        x = self.normalization(x)

        x = self.dropout(
            x,
            training=training
        )

        return x


# ============================================================
# 8. ADAPTIVE MULTIMODAL FUSION
# ============================================================

class AdaptiveMultimodalFusion(
    layers.Layer
):
    """
    Adaptive gated fusion of MRI and clinical embeddings.
    """

    def __init__(
        self,
        fusion_dim=FUSION_DIM,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.image_projection = layers.Dense(
            fusion_dim,
            activation="relu"
        )

        self.clinical_projection = layers.Dense(
            fusion_dim,
            activation="relu"
        )

        self.gate = layers.Dense(
            fusion_dim,
            activation="sigmoid"
        )

        self.output_projection = layers.Dense(
            fusion_dim,
            activation="relu"
        )

    def call(
        self,
        image_features,
        clinical_features
    ):

        image_features = (
            self.image_projection(
                image_features
            )
        )

        clinical_features = (
            self.clinical_projection(
                clinical_features
            )
        )

        combined = tf.concat(
            [
                image_features,
                clinical_features
            ],
            axis=-1
        )

        gate = self.gate(
            combined
        )

        fused = (
            gate * image_features
            +
            (1.0 - gate)
            * clinical_features
        )

        fused = self.output_projection(
            fused
        )

        return fused


# ============================================================
# 9. BREASTFUSION-NET
# ============================================================

def build_breastfusion_net(
    clinical_dim=CLINICAL_DIM,
    learning_rate=1e-4
):
    """
    Complete BreastFusion-Net architecture.
    """

    # --------------------------------------------------------
    # MRI INPUT
    # --------------------------------------------------------

    image_input = layers.Input(
        shape=(
            IMG_SIZE,
            IMG_SIZE,
            IMAGE_CHANNELS
        ),
        name="mri_input"
    )

    # --------------------------------------------------------
    # CLINICAL INPUT
    # --------------------------------------------------------

    clinical_input = layers.Input(
        shape=(clinical_dim,),
        name="clinical_input"
    )

    # --------------------------------------------------------
    # RESNET50
    # --------------------------------------------------------

    resnet = build_resnet50_backbone(
        trainable=False
    )

    spatial_features = resnet(
        image_input
    )

    # --------------------------------------------------------
    # S-TRANSFORMER
    # --------------------------------------------------------

    contextual_features = STransformer(
        embed_dim=2048,
        num_heads=8,
        ff_dim=4096
    )(
        spatial_features
    )

    # --------------------------------------------------------
    # CHANNEL-WISE REFINEMENT
    # --------------------------------------------------------

    refined_features = (
        ChannelWiseRefinement()
        (contextual_features)
    )

    # --------------------------------------------------------
    # GLOBAL FEATURE EMBEDDING
    # --------------------------------------------------------

    image_embedding = (
        GlobalFeatureEmbedding(
            embedding_dim=IMAGE_EMBED_DIM
        )
        (refined_features)
    )

    # --------------------------------------------------------
    # CFLANN CLINICAL BRANCH
    # --------------------------------------------------------

    clinical_embedding = (
        ClinicalFeatureEmbedding(
            embedding_dim=CLINICAL_EMBED_DIM
        )
        (clinical_input)
    )

    # --------------------------------------------------------
    # ADAPTIVE FUSION
    # --------------------------------------------------------

    fused_features = (
        AdaptiveMultimodalFusion(
            fusion_dim=FUSION_DIM
        )
        (
            image_embedding,
            clinical_embedding
        )
    )

    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    x = layers.Dropout(
        0.30
    )(fused_features)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.20
    )(x)

    output = layers.Dense(
        1,
        activation="sigmoid",
        name="malignancy_probability"
    )(x)

    model = Model(
        inputs=[
            image_input,
            clinical_input
        ],
        outputs=output,
        name="BreastFusion-Net"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )

    return model


# ============================================================
# 10. GREY WOLF OPTIMIZATION
# ============================================================

class GreyWolfOptimizer:
    """
    Lightweight GWO implementation for hyperparameter search.

    Parameters optimized in this implementation:
        - learning rate
        - dropout
        - batch size

    The objective function must perform validation on the
    training/development partition only.
    """

    def __init__(
        self,
        objective_function,
        bounds,
        population_size=5,
        iterations=5
    ):

        self.objective_function = (
            objective_function
        )

        self.bounds = np.asarray(
            bounds,
            dtype=np.float32
        )

        self.population_size = (
            population_size
        )

        self.iterations = iterations

        self.dimension = len(bounds)

    def initialize_population(self):

        population = []

        for _ in range(
            self.population_size
        ):

            wolf = []

            for low, high in self.bounds:

                wolf.append(
                    np.random.uniform(
                        low,
                        high
                    )
                )

            population.append(wolf)

        return np.asarray(
            population,
            dtype=np.float32
        )

    def optimize(self):

        population = (
            self.initialize_population()
        )

        alpha = None
        beta = None
        delta = None

        alpha_score = np.inf
        beta_score = np.inf
        delta_score = np.inf

        for iteration in range(
            self.iterations
        ):

            for wolf in population:

                score = (
                    self.objective_function(
                        wolf
                    )
                )

                if score < alpha_score:

                    delta_score = beta_score
                    delta = beta

                    beta_score = alpha_score
                    beta = alpha

                    alpha_score = score
                    alpha = wolf.copy()

                elif score < beta_score:

                    delta_score = beta_score
                    delta = beta

                    beta_score = score
                    beta = wolf.copy()

                elif score < delta_score:

                    delta_score = score
                    delta = wolf.copy()

            a = 2 - (
                2 * iteration
                / max(
                    self.iterations - 1,
                    1
                )
            )

            for i in range(
                self.population_size
            ):

                for j in range(
                    self.dimension
                ):

                    r1 = np.random.random()
                    r2 = np.random.random()

                    A1 = (
                        2 * a * r1 - a
                    )

                    C1 = 2 * r2

                    D_alpha = abs(
                        C1 * alpha[j]
                        - population[i, j]
                    )

                    X1 = (
                        alpha[j]
                        - A1 * D_alpha
                    )

                    # Beta
                    if beta is not None:

                        r1 = np.random.random()
                        r2 = np.random.random()

                        A2 = (
                            2 * a * r1 - a
                        )

                        C2 = 2 * r2

                        D_beta = abs(
                            C2 * beta[j]
                            - population[i, j]
                        )

                        X2 = (
                            beta[j]
                            - A2 * D_beta
                        )

                    else:
                        X2 = X1

                    # Delta
                    if delta is not None:

                        r1 = np.random.random()
                        r2 = np.random.random()

                        A3 = (
                            2 * a * r1 - a
                        )

                        C3 = 2 * r2

                        D_delta = abs(
                            C3 * delta[j]
                            - population[i, j]
                        )

                        X3 = (
                            delta[j]
                            - A3 * D_delta
                        )

                    else:
                        X3 = X1

                    population[i, j] = (
                        X1 + X2 + X3
                    ) / 3.0

                    population[i, j] = np.clip(
                        population[i, j],
                        self.bounds[j, 0],
                        self.bounds[j, 1]
                    )

        return alpha, alpha_score


# ============================================================
# 11. PATIENT-WISE STRATIFIED K-FOLD EVALUATION
# ============================================================

def patient_wise_kfold_evaluation(
    images,
    clinical_data,
    labels,
    n_splits=5,
    epochs=10,
    batch_size=16
):
    """
    Perform patient-wise stratified K-fold evaluation.

    IMPORTANT:
        Each row must correspond to one patient.

    Returns:
        Fold-wise metrics and pooled out-of-fold predictions.
    """

    images = np.asarray(images)
    clinical_data = np.asarray(
        clinical_data,
        dtype=np.float32
    )
    labels = np.asarray(
        labels,
        dtype=np.int32
    )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    oof_true = []
    oof_pred = []
    oof_prob = []

    fold_results = []

    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        skf.split(
            images,
            labels
        ),
        start=1
    ):

        print(
            f"\n========== Fold {fold}/{n_splits} =========="
        )

        # ----------------------------------------------------
        # Patient-wise partition
        # ----------------------------------------------------

        X_train_img = images[
            train_idx
        ]

        X_val_img = images[
            val_idx
        ]

        X_train_clinical = clinical_data[
            train_idx
        ]

        X_val_clinical = clinical_data[
            val_idx
        ]

        y_train = labels[
            train_idx
        ]

        y_val = labels[
            val_idx
        ]

        # ----------------------------------------------------
        # Build fresh model for each fold
        # ----------------------------------------------------

        model = build_breastfusion_net()

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.fit(
            [
                X_train_img,
                X_train_clinical
            ],
            y_train,
            validation_data=(
                [
                    X_val_img,
                    X_val_clinical
                ],
                y_val
            ),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # ----------------------------------------------------
        # Out-of-fold prediction
        # ----------------------------------------------------

        probabilities = model.predict(
            [
                X_val_img,
                X_val_clinical
            ],
            verbose=0
        ).ravel()

        predictions = (
            probabilities >= 0.5
        ).astype(np.int32)

        # ----------------------------------------------------
        # Store OOF predictions
        # ----------------------------------------------------

        oof_true.extend(
            y_val.tolist()
        )

        oof_pred.extend(
            predictions.tolist()
        )

        oof_prob.extend(
            probabilities.tolist()
        )

        # ----------------------------------------------------
        # Fold metrics
        # ----------------------------------------------------

        fold_accuracy = accuracy_score(
            y_val,
            predictions
        )

        fold_precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        fold_recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        fold_f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        fold_auc = roc_auc_score(
            y_val,
            probabilities
        )

        fold_results.append({
            "fold": fold,
            "accuracy": fold_accuracy,
            "precision": fold_precision,
            "recall": fold_recall,
            "f1": fold_f1,
            "auc": fold_auc
        })

    # ========================================================
    # POOLED OUT-OF-FOLD METRICS
    # ========================================================

    oof_true = np.asarray(
        oof_true
    )

    oof_pred = np.asarray(
        oof_pred
    )

    oof_prob = np.asarray(
        oof_prob
    )

    pooled_accuracy = accuracy_score(
        oof_true,
        oof_pred
    )

    pooled_precision = precision_score(
        oof_true,
        oof_pred,
        zero_division=0
    )

    pooled_recall = recall_score(
        oof_true,
        oof_pred,
        zero_division=0
    )

    pooled_f1 = f1_score(
        oof_true,
        oof_pred,
        zero_division=0
    )

    pooled_auc = roc_auc_score(
        oof_true,
        oof_prob
    )

    pooled_cm = confusion_matrix(
        oof_true,
        oof_pred
    )

    print("\n======================================")
    print("POOLED OUT-OF-FOLD RESULTS")
    print("======================================")

    print(
        f"Accuracy  : {pooled_accuracy:.4f}"
    )

    print(
        f"Precision : {pooled_precision:.4f}"
    )

    print(
        f"Recall    : {pooled_recall:.4f}"
    )

    print(
        f"F1-score  : {pooled_f1:.4f}"
    )

    print(
        f"AUC       : {pooled_auc:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        pooled_cm
    )

    return {
        "fold_results": fold_results,
        "y_true": oof_true,
        "y_pred": oof_pred,
        "y_probability": oof_prob,
        "confusion_matrix": pooled_cm,
        "accuracy": pooled_accuracy,
        "precision": pooled_precision,
        "recall": pooled_recall,
        "f1": pooled_f1,
        "auc": pooled_auc
    }


# ============================================================
# 12. GRAD-CAM
# ============================================================

def make_gradcam_heatmap(
    image,
    clinical_vector,
    model,
    last_conv_layer_name,
    pred_index=None
):
    """
    Generate Grad-CAM heatmap for the MRI branch.

    Grad-CAM is calculated with respect to the final convolutional
    feature representation before global feature embedding.

    Parameters:
        image:
            Single MRI image with shape (224,224,3)

        clinical_vector:
            Corresponding clinical feature vector

        model:
            Trained BreastFusion-Net

        last_conv_layer_name:
            Name of convolutional layer used for Grad-CAM
    """

    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(
                last_conv_layer_name
            ).output,
            model.output
        ]
    )

    image_tensor = tf.convert_to_tensor(
        image[None, ...],
        dtype=tf.float32
    )

    clinical_tensor = tf.convert_to_tensor(
        clinical_vector[None, ...],
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = (
            grad_model(
                [
                    image_tensor,
                    clinical_tensor
                ]
            )
        )

        if pred_index is None:
            class_channel = predictions[:, 0]
        else:
            class_channel = predictions[
                :,
                pred_index
            ]

    gradients = tape.gradient(
        class_channel,
        conv_outputs
    )

    # Global average pooling of gradients
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = (
        pooled_gradients[0]
    )

    # Weight convolutional feature maps
    heatmap = tf.reduce_sum(
        conv_outputs
        * pooled_gradients[None, None, :],
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = (
        heatmap
        / (max_value + 1e-8)
    )

    return heatmap.numpy()


# ============================================================
# 13. GRAD-CAM OVERLAY
# ============================================================

def overlay_gradcam(
    image,
    heatmap,
    alpha=0.4
):
    """
    Create a Grad-CAM overlay.
    """

    import cv2

    image = np.asarray(
        image
    ).astype(np.uint8)

    heatmap = cv2.resize(
        heatmap,
        (
            image.shape[1],
            image.shape[0]
        )
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    if image.shape[-1] == 1:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        colored_heatmap,
        alpha,
        0
    )

    return overlay


# ============================================================
# 14. SAVE OF RESULTS
# ============================================================

def save_oof_predictions(
    results,
    output_path="results/oof_predictions.npz"
):

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True
    )

    np.savez(
        output_path,
        y_true=results["y_true"],
        y_pred=results["y_pred"],
        y_probability=results[
            "y_probability"
        ]
    )

    print(
        f"OOF predictions saved to: "
        f"{output_path}"
    )


