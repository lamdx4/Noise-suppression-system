import os
import shutil
import random

def copy_audio_file(source_path: str, destination_path: str):
    """
    Sao chép một file âm thanh từ source_path đến destination_path.

    :param source_path: Đường dẫn file nguồn.
    :param destination_path: Đường dẫn file đích.
    """
    try:
        shutil.copy2(source_path, destination_path)
        print(f"Đã sao chép file thành công từ {source_path} đến {destination_path}")
    except Exception as e:
        print(f"Lỗi khi sao chép file: {e}")


def take_voice_dataset():
    """
    Hàm này sẽ sao chép các file âm thanh từ thư mục output_audio vào thư mục chứa dataset.
    """
    lst_audio =[]
    for root, folder, files in os.walk("audio_datasets"):
        for file in files:
            source_path = os.path.join(root, file)
            lst_audio.append(source_path)
    random_shuffled = random.shuffle(lst_audio)
    lst_audio = lst_audio[:2000]
    if not os.path.exists("datasets/clean"):
        os.makedirs("datasets/clean")
    for i in range(0, len(lst_audio)):
        destination_path = f"datasets/clean/{i}.wav"
        copy_audio_file(lst_audio[i], destination_path)
def take_noise_dataset():
    """
    Hàm này sẽ sao chép các file âm thanh từ thư mục output_audio vào thư mục chứa dataset.
    """

    lst_audio =[]
    take_list = [16,22,23]
    name_list =["wind","clap","breath"]
    index_list =[0,0,0]
    if not os.path.exists("datasets/noise"):
        os.makedirs("datasets/noise")
    for file in os.listdir("audio"):
        tpe = int(file.split(".")[0].split("-")[-1])
        if tpe in take_list:
            source_path = os.path.join("audio", file)
            save_path = f"datasets/noise/{name_list[take_list.index(tpe)]}-{index_list[take_list.index(tpe)]}.wav"
            copy_audio_file(source_path, save_path)
            index_list[take_list.index(tpe)] += 1
if __name__ == "__main__":
    take_noise_dataset()




