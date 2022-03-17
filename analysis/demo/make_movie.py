# dependencies: imageio, fpdf2, wand, ffmpeg

import pandas as pd
import imageio
from fpdf import FPDF
from pathlib import Path
from wand.image import Image
    
data = pd.read_csv('12TokenSents.csv')

# we want to create a movie for each 'source' sentence
for group_name, group in data.groupby('sentence_num') :

    # make a directory for this sentence
    Path("sent{}".format(group_name)).mkdir(parents=True, exist_ok=True)
    num_substeps = int(len(group['substep']) / 10)
    filenames = []

    # initialize a pdf
    pdf = FPDF(format=(200,150))
    pdf_filename = "sent{}/steps.pdf".format(group_name)
    pdf.add_font('DejaVu', '', 'DejaVuSansMono.ttf', uni=True)
    pdf.set_font('DejaVu', '', 14)

    # Make a page for each 'substep'
    for substep in range(num_substeps) :
        sub_d = group.iloc[substep * 10 : (substep * 10) + 10]
        pdf.add_page()

        # write the step at the top...
        pdf.cell(200, 10, txt='epoch:{}, step:{}'.format(
            str(sub_d['iter_num'].iloc[1]), str(sub_d['substep'].iloc[1])
        ))
        pdf.ln(10)

        # write a sentence on each line
        for sentence in sub_d['sentence'] :
            pdf.cell(200, 10, txt=sentence)
            pdf.ln(10)
    pdf.output(pdf_filename)

    print('converting pages of pdf to pngs...')
    images = []
    all_pages = Image(filename=pdf_filename) 
    for i, page in enumerate(all_pages.sequence):
        with Image(page) as img:
            img.format = 'png'
            img.save(filename=pdf_filename+'.png')
            images.append(imageio.imread(pdf_filename+'.png'))

    print('creating movie from pngs...')
    imageio.mimwrite('sent{}/movie.mp4'.format(group_name), images, fps=4)
