from PIL import Image
import numpy as np


def timestamp_to_filename(time_stamp_ms: int) -> str:
    return f'{time_stamp_ms:015d}'


def store_image(data: np.ndarray, file_name: str) -> None:
    assert np.amin(data) >= 0
    assert np.amax(data) <= 1
    assert data.dtype == np.float32 or data.dtype == np.float16 or data.dtype == np.float64

    processed_data = (data * 255).astype(np.uint8)
    im = Image.fromarray(processed_data)
    im.save(file_name)  # await


def store_array_as_numpy(data: np.ndarray, file_name: str) -> None:
    np.save(file_name, data)


def store_array_to_file(data: np.ndarray, file_name: str, time_stamp: int = 0) -> None:
    message = f'{time_stamp} : ' + ' '.join(f'{x:0.5f}' for x in data) + '\n'
    with open(file_name, 'a') as f:
        f.write(message)
