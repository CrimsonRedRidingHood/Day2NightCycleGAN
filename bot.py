import os
import asyncio
from aiogram import Bot, Dispatcher, types, filters
from aiogram.filters import CommandStart
from aiogram.filters import Command
from aiogram.types import FSInputFile
from aiogram import F

import cycle_gan_model

bot = Bot(token=open('token.txt').read())
dp = Dispatcher()

initial_image_filename = 'initial_image'
result_image_filename = 'result_image.jpg'

model = []

@dp.message(CommandStart())
async def cmd_handler(message: types.Message):
    await message.answer('CycleGAN bot. Please, send an image')
    
@dp.message(Command('help'))
async def cmd_handler(message: types.Message):
    await message.answer('CycleGAN bot. Please, send an image')
    
@dp.message(F.photo)
async def compressed_handler(message: types.Message, bot: bot):
    await bot.download(
        message.photo[-1],
        destination=initial_image_filename
    )
    
    await message.answer('Processing...')
    
    cycle_gan_model.process_image(model, initial_image_filename, result_image_filename)
    await message.answer_photo(FSInputFile(result_image_filename))

@dp.message(F.text)
async def text_messge_handler(message: types.Message):
    await message.answer("Unknown request. Please, send an image")

@dp.message(F.document)
async def document_handler(message: types.Message):
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    await bot.download_file(file.file_path, initial_image_filename)
    
    await message.answer('Processing...')
    
    cycle_gan_model.process_image(model, initial_image_filename, result_image_filename)
    await message.answer_photo(FSInputFile(result_image_filename))

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    model = cycle_gan_model.load_model()
    asyncio.run(main())