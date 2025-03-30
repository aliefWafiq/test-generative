import tensorflow as tf
import numpy as np
import librosa
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from tensorflow.io import gfile



SPEECH_DATA = 'speech_data'
EXPECTED_SAMPLES = 16000
NOISE_FLOOR = 0.1
MINIMUM_VOICE_LENGTH = EXPECTED_SAMPLES / 4

words = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
    'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
    'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes',
    'zero', '_background',
]

def get_files(word):
    return gfile.glob(SPEECH_DATA + '/' + word + '/*.wav')

def get_voice_position(audio, noise_floor):
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    start, end = librosa.effects.trim(audio, top_db=noise_floor)
    return int(start), int(end)

def get_voice_length(audio, noise_floor):
    position = get_voice_position(audio, noise_floor)
    return position[1] - position[0]

def is_voice_present(audio, noise_floor, required_length):
    voice_length = get_voice_length(audio, noise_floor)
    return voice_length >= required_length

def is_correct_length(audio, expected_length):
    return audio.shape[0] == expected_length

def is_valid_file(file_name):
    audio, _ = librosa.load(file_name, sr=None)
    if not is_correct_length(audio, EXPECTED_SAMPLES):
        return False
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    if not is_voice_present(audio, NOISE_FLOOR, MINIMUM_VOICE_LENGTH):
        return False
    return True

def get_spectrogram(audio):
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=EXPECTED_SAMPLES)

    spectrogram = np.log10(spectrogram + 1e-6)
    return spectrogram

def process_file(file_path):
    audio, _ = librosa.load(file_path, sr=None)
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    voice_start, voice_end = get_voice_position(audio, NOISE_FLOOR)
    end_gap = len(audio) - voice_end
    random_offset = np.random.uniform(0, voice_start + end_gap)
    audio = np.roll(audio, int(-random_offset + end_gap))

    return get_spectrogram(audio)

train = []
validate = []
test = []

TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1

def process_files(file_names, label, repeat=1):
    file_names = tf.repeat(file_names, repeat).numpy()
    return [(process_file(file_name), label) for file_name in tqdm(file_names, desc=f"{word} ({label})", leave=False)]

def process_word(word, repeat=1):
    label = words.index(word)
    file_names = [file_name for file_name in tqdm(get_files(word), desc="Checking", leave=False) if is_valid_file(file_name)]
    np.random.shuffle(file_names)
    train_size = int(TRAIN_SIZE * len(file_names))
    validation_size = int(VALIDATION_SIZE * len(file_names))
    test_size = int(TEST_SIZE * len(file_names))
    train.extend(process_files(file_names[:train_size], label, repeat=repeat))
    validate.extend(process_files(file_names[train_size:train_size + validation_size], label, repeat=repeat))
    test.extend(process_files(file_names[train_size + validation_size:], label, repeat=repeat))

for word in tqdm(words, desc="Processing words"):
    if '_' not in word:
        repeat = 70 if word == 'marvin' else 1
        process_word(word, repeat=repeat)

print(len(train), len(test), len(validate))

# Save the computed data
np.savez_compressed("training_spectrogram.npz", X=[x[0] for x in train], Y=[x[1] for x in train])
print("Saved training data")
np.savez_compressed("validation_spectrogram.npz", X=[x[0] for x in validate], Y=[x[1] for x in validate])
print("Saved validation data")
np.savez_compressed("test_spectrogram.npz", X=[x[0] for x in test], Y=[x[1] for x in test])
print("Saved test data")

def process_background(file_name, label):
    # load the audio file
    audio, _ = librosa.load(file_name, sr=None)

    audio_length = len(audio)
    samples = []
    for section_start in tqdm(range(0, audio_length - EXPECTED_SAMPLES, 8000), desc=file_name, leave=False):
        section_end = section_start + EXPECTED_SAMPLES
        section = audio[section_start:section_end]
        # get the spectrogram
        spectrogram = get_spectrogram(section)
        samples.append((spectrogram, label))

    # simulate random utterances
    for section_index in tqdm(range(1000), desc="Simulated Words", leave=False):
        section_start = np.random.randint(0, audio_length - EXPECTED_SAMPLES)
        section_end = section_start + EXPECTED_SAMPLES
        section = np.reshape(audio[section_start:section_end], (EXPECTED_SAMPLES))

        result = np.zeros((EXPECTED_SAMPLES))
        # create a pseudo bit of voice
        voice_length = np.random.randint(MINIMUM_VOICE_LENGTH/2, EXPECTED_SAMPLES)
        voice_start = np.random.randint(0, EXPECTED_SAMPLES - voice_length)
        hamming = np.hamming(voice_length)
        # amplify the voice section
        result[voice_start:voice_start+voice_length] = hamming * section[voice_start:voice_start+voice_length]
        # get the spectrogram
        spectrogram = get_spectrogram(np.reshape(section, (16000, 1)))
        samples.append((spectrogram, label))
        
    np.random.shuffle(samples)
    
    train_size=int(TRAIN_SIZE*len(samples))
    validation_size=int(VALIDATION_SIZE*len(samples))
    test_size=int(TEST_SIZE*len(samples))
    
    train.extend(samples[:train_size])
    validate.extend(samples[train_size:train_size+validation_size])
    test.extend(samples[train_size+validation_size:])

for file_name in tqdm(get_files('_background_noise_'), desc="Processing Background Noise"):
    process_background(file_name, words.index("_background"))
    
print(len(train), len(test), len(validate))

def process_problem_noise(file_name, label):
    samples = []
    # load the audio file
    audio, _ = librosa.load(file_name, sr=None)

    audio_length = len(audio)
    for section_start in tqdm(range(0, audio_length - EXPECTED_SAMPLES, 400), desc=file_name, leave=False):
        section_end = section_start + EXPECTED_SAMPLES
        section = audio[section_start:section_end]
        # get the spectrogram
        spectrogram = get_spectrogram(section)
        samples.append((spectrogram, label))
        
    np.random.shuffle(samples)
    
    train_size=int(TRAIN_SIZE*len(samples))
    validation_size=int(VALIDATION_SIZE*len(samples))
    test_size=int(TEST_SIZE*len(samples))
    
    train.extend(samples[:train_size])
    validate.extend(samples[train_size:train_size+validation_size])
    test.extend(samples[train_size+validation_size:])

for file_name in tqdm(get_files("_problem_noise_"), desc="Processing problem noise"):
    process_problem_noise(file_name, words.index("_background"))

def process_mar_sounds(file_name, label):
    samples = []
    # load the audio file
    audio, _ = librosa.load(file_name, sr=None)

    audio_length = len(audio)
    for section_start in tqdm(range(0, audio_length - EXPECTED_SAMPLES, 4000), desc=file_name, leave=False):
        section_end = section_start + EXPECTED_SAMPLES
        section = audio[section_start:section_end]
        section = section - np.mean(section)
        section = section / np.max(np.abs(section))
        # add some random background noise
        background_volume = np.random.uniform(0, 0.1)
        # get the background noise files
        background_files = get_files('_background_noise_')
        background_file = np.random.choice(background_files)
        background, _ = librosa.load(background_file, sr=None)
        background_start = np.random.randint(0, len(background) - 16000)
        # normalise the background noise
        background = background - np.mean(background)
        background = background / np.max(np.abs(background))
        # mix the audio with the scaled background
        section = section + background_volume * background
        # get the spectrogram
        spectrogram = get_spectrogram(section)
        samples.append((spectrogram, label))
        
    np.random.shuffle(samples)
    
    train_size=int(TRAIN_SIZE*len(samples))
    validation_size=int(VALIDATION_SIZE*len(samples))
    test_size=int(TEST_SIZE*len(samples))
    
    train.extend(samples[:train_size])
    validate.extend(samples[train_size:train_size+validation_size])
    test.extend(samples[train_size+validation_size:])

for file_name in tqdm(get_files("_mar_sounds_"), desc="Processing problem noise"):
    process_mar_sounds(file_name, words.index("_background"))

np.savez_compressed(
    "training_spectrogram.npz",
    X=X_train, Y=Y_train)
print("Saved training data")
np.savez_compressed(
    "validation_spectrogram.npz",
    X=X_validate, Y=Y_validate)
print("Saved validation data")
np.savez_compressed(
    "test_spectrogram.npz",
    X=X_test, Y=Y_test)
print("Saved test data")

# get the width and height of the spectrogram "image"
IMG_WIDTH=X_train[0].shape[0]
IMG_HEIGHT=X_train[0].shape[1]

def plot_images2(images_arr, imageWidth, imageHeight):
    fig, axes = plt.subplots(5, 5, figsize=(10, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(np.reshape(img, (imageWidth, imageHeight)))
        ax.axis("off")
    plt.tight_layout()
    plt.show()

word_index = words.index("marvin")

X_marvins = np.array(X_train)[np.array(Y_train) == word_index]
Y_marvins = np.array(Y_train)[np.array(Y_train) == word_index]
plot_images2(X_marvins[:20], IMG_WIDTH, IMG_HEIGHT)
print(Y_marvins[:20])
