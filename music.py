import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

class Config:
    """Project configuration parameters."""
    DATASET_DIR = Path("GTZAN_student")
    NEW_AUDIO = Path("my_music.wav")
    SAMPLE_RATE = 22050
    GENRES = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock"
    ]

class AudioFeatureExtractor:
    """Class for extracting numerical features from audio files."""
    
    @staticmethod
    def extract_features(file_path: Path) -> np.ndarray:
        # Load audio
        audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE, mono=True)

        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        centroid_mean = np.mean(spectral_centroid)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        rolloff_mean = np.mean(spectral_rolloff)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        tempo = float(np.asarray(tempo).flatten()[0])

        # Combine all features into a vector
        features = np.concatenate([
            mfcc_mean, mfcc_std, [centroid_mean], [bandwidth_mean],
            [rolloff_mean], [zcr_mean], [tempo]
        ])
        return features

class MusicGenreClassifier:
    """Class for model training, evaluation, and genre prediction."""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        print("Training complete!")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "=" * 60)
        print("CLASSIFICATION RESULTS")
        print("=" * 60)
        print(f"\nModel accuracy: {accuracy:.2%}")
        print("\nDetailed report:")
        print(classification_report(y_test, y_pred, labels=Config.GENRES, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=Config.GENRES)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Config.GENRES)
        disp.plot(xticks_rotation=45)
        plt.title("Music Genre Classification Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def predict_new_audio(self, file_path: Path):
        print("\n" + "=" * 60)
        print("CLASSIFICATION OF NEW AUDIO")
        print("=" * 60)
        
        if not file_path.exists():
            print(f"File {file_path.name} not found.")
            return

        print(f"File: {file_path.name}")
        features = AudioFeatureExtractor.extract_features(file_path).reshape(1, -1)
        
        predicted_genre = self.model.predict(features)[0]
        print(f"\nPredicted genre: {predicted_genre.upper()}")
        
        probabilities = self.model.predict_proba(features)[0]
        print("\nProbabilities:")
        results = sorted(zip(self.model.classes_, probabilities), key=lambda x: x[1], reverse=True)
        for genre, prob in results:
            print(f"{genre:12s}: {prob:.2%}")

class Application:
    """Main class for orchestrating the process."""
    
    def __init__(self):
        self.X = []
        self.y = []
        self.classifier = MusicGenreClassifier()

    def load_dataset(self):
        print("=" * 60)
        print("LOADING AUDIO FILES")
        print("=" * 60)
        
        if not Config.DATASET_DIR.exists():
            raise FileNotFoundError(f"Folder {Config.DATASET_DIR.name} not found in the project root.")
        for genre in Config.GENRES:
            genre_path = Config.DATASET_DIR / genre
            if not genre_path.exists():
                print(f"WARNING: folder {genre} not found!")
                continue

            audio_files = list(genre_path.glob("*.wav"))
            for file_path in audio_files:
                try:
                    features = AudioFeatureExtractor.extract_features(file_path)
                    self.X.append(features)
                    self.y.append(genre)
                except Exception as e:
                    print(f"Error in file {file_path.name}: {e}")

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        print("\nDATASET FORMED")
        print(f"Number of audio files: {len(self.X)}")
        print(f"Number of features per file: {self.X.shape[1]}")

    def run(self):
        self.load_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.20, random_state=42, stratify=self.y
        )
        print(f"\nTraining set: {len(X_train)} files")
        print(f"Test set: {len(X_test)} files")
        
        # Training and evaluation
        self.classifier.train(X_train, y_train)
        self.classifier.evaluate(X_test, y_test)
        
        # New track classification
        self.classifier.predict_new_audio(Config.NEW_AUDIO)

if __name__ == "__main__":
    app = Application()
    app.run()