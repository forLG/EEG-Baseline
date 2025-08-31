import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# 1. CNN Feature Extractor Block
# -----------------------------------------------------------------------------
# As described: "The CNN architecture consists of 5 convolution layers"
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        # Note: PyTorch Conv1d expects input as (N, C_in, L_in)
        self.cnn_layers = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Layer 2
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Layer 3
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Layer 4
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Layer 5
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.cnn_layers(x)
        return self.flatten(x)

# -----------------------------------------------------------------------------
# 2. Transformer Encoder Block
# -----------------------------------------------------------------------------
# As described: "deploy a transformer encoder to identify patterns in the features extracted by the CNN"
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        # Multi-Head Attention Layer
        self.att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        # Layer Normalization
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        # Dropout
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x):
        # As stated: "the local features extracted by the CNN are the query,
        # key, and value simultaneously."
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        # Residual connection 1
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        # Residual connection 2
        return self.layernorm2(out1 + ffn_output)
        
# -----------------------------------------------------------------------------
# 3. Main CNN-Transformer Model
# -----------------------------------------------------------------------------
class CNNTransformer(nn.Module):
    """
    The full CNN-Transformer model for EEG artifact detection as described
    in the paper, implemented in PyTorch.
    """
    def __init__(self, n_channels=19, L_seconds=3, fs=128, n_classes=2):
        super(CNNTransformer, self).__init__()
        self.n_channels = n_channels
        
        # ----- A. Input Segmentation Parameters -----
        # As described: "split into 0.5s local segments with 25% overlap"
        self.segment_length = int(0.5 * fs) # 0.5s * 128Hz = 64 samples
        overlap = 0.25
        self.step = int(self.segment_length * (1 - overlap)) # 64 * 0.75 = 48 samples
        
        # ----- B. CNN Feature Extractor -----
        self.cnn_extractor = CNNFeatureExtractor()
        
        # Calculate the feature dimension after CNN processing
        # This is needed to initialize the subsequent layers correctly.
        # We can do this by a dummy forward pass.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.segment_length)
            cnn_output_dim = self.cnn_extractor(dummy_input).shape[1]

        transformer_embed_dim = cnn_output_dim * self.n_channels
        
        # ----- C. Transformer Encoder -----
        num_transformer_heads = 8
        ff_dim_transformer = 1024
        self.transformer_encoder = TransformerEncoder(
            embed_dim=transformer_embed_dim,
            num_heads=num_transformer_heads,
            ff_dim=ff_dim_transformer
        )
        
        # ----- D. Classifier Head -----
        self.flatten = nn.Flatten()
        
        # As described: "Two fully connected (FC) layers containing 100 and 2 neurons"
        # and "Before the final FC layer, we include a dropout layer with a probability of 0.5"
        self.classifier = nn.Sequential(
            nn.Linear(cnn_output_dim, 100), # The input features to FC depends on Transformer output, but it's concatenated later. We'll handle this in forward pass.
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, n_classes)
        )
        
        # We need to dynamically create the first linear layer of the classifier
        # because its input size depends on the number of segments.
        # We'll calculate it in the forward pass for the first time or set it here if L is fixed.
        input_length_samples = L_seconds * fs
        num_segments = (input_length_samples - self.segment_length) // self.step + 1
        classifier_input_dim = num_segments * transformer_embed_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, n_classes)
        )


    def forward(self, x):
        # 1. Segmentation
        # PyTorch expects (N, C, L)
        # The output will be (batch_size, n_channels, num_segments, segment_length)
        segments = x.unfold(dimension=2, size=self.segment_length, step=self.step)
        
        batch_size, n_channels, num_segments, segment_length = segments.shape
        
        # Reshape to treat all segments as a single batch for efficient CNN processing
        # (batch_size * num_segments, 1, segment_length)
        segments_reshaped = segments.permute(0, 2, 1, 3).reshape(-1, 1, segment_length)
        
        # 2. CNN Feature Extraction
        cnn_features = self.cnn_extractor(segments_reshaped)
        
        # 3. Reshape back to sequence for Transformer
        # (batch_size, num_segments, cnn_output_features)
        cnn_features_sequence = cnn_features.view(batch_size, num_segments, -1)
        
        # 4. Transformer Encoder
        refined_features = self.transformer_encoder(cnn_features_sequence)
        
        # 5. Classifier Head
        # Flatten the sequence of features ("Concatenation")
        concatenated_features = self.flatten(refined_features)
        
        # Get final prediction
        output = self.classifier(concatenated_features)
        
        return output


# Example usage
if __name__ == '__main__':
    N_CHANNELS = 19
    BATCH_SIZE = 4
    L_SECONDS = 10
    FS = 200
    
    model = CNNTransformer(n_channels=N_CHANNELS, L_seconds=L_SECONDS, fs=FS)

    window_length = L_SECONDS * FS
    dummy_input = torch.randn(BATCH_SIZE, N_CHANNELS, window_length)

    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"input shape {dummy_input.shape}")
    print(f"output shape {output.shape}")# Should be (BATCH_SIZE, 2)