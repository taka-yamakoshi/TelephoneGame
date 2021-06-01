import pandas as pd
import imageio
from fpdf import FPDF
from pathlib import Path
from wand.image import Image
    
data = pd.read_csv('12TokenSents.csv')

# write to file a pdf for each 'source' sentence
for group_name, group in data.groupby('sentence_num') :
    num_substeps = int(len(group['substep']) / 10)
    filenames = []
    Path("sent{}".format(group_name)).mkdir(parents=True, exist_ok=True)

    print('generating pdf')
    pdf = FPDF(format=(200,150))
    pdf.add_font('DejaVu', '', 'DejaVuSansMono.ttf', uni=True)
    pdf.set_font('DejaVu', '', 14)

    # Make a page for each 'substep'
    for substep in range(num_substeps) :
        sub_d = group.iloc[substep * 10 : (substep * 10) + 10]
        pdf.add_page()

        # write the step at the top...
        pdf.cell(200, 10, txt='epoch:{}, step:{}'.format(
            str(sub_d['iter_num'][0]), str(sub_d['substep'][0])
        ))
        pdf.ln(10)

        # write a sentence on each line
        for sentence in sub_d['sentence'] :
            pdf.cell(200, 10, txt=sentence)
            pdf.ln(10)
    f = "sent{}.pdf".format(group_name, substep)
    pdf.output(f)

    print('converting to pngs...')
    images = []
    all_pages = Image(filename=f) 
    for i, page in enumerate(all_pages.sequence):
        print(i)
        with Image(page) as img:
            img.format = 'png'
            img.save(filename=f+'.png')
            images.append(imageio.imread(f+'.png'))

    print('creating gif...')
    imageio.mimwrite('movie-{}.mp4'.format(group_name), images, fps=2)    
