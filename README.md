# Day2NightCycleGAN
A CycleGAN implementation. Used on Day vs Night dataset.

Для работы бота понадобится питон aiogram - установить можно с помощью команды pip install aiogram.
Для запуска бота необходимо, во-первых, поместить файл с нейросетью в папку с репозиторием и назвать его `gen.nn` (это имя можно изменить в cycle_gan_model.py), во-вторых, в файлу cycle_gan_model.py изменить параметр IMAGE_SIZE на соответствующий размеру изображений, на которых тренировалась нейросеть, в-третьих, необходимо в эту же папку поместить файл `token.txt`, в котором сохранить свой токен для бота. После чего можно запускать бота с помощью `python bot.py` и переходить в него, скармливая ему картинки. Для большего качества изображений рекомендуется использовать нейросеть, натренированную на изображениях размером 256х256 и больше.

Детали проекта описаны в начале ноутбука.
