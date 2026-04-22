'''20240620 initial template to be modified accordingly'''
import os

def generate_slide_frames(title, filenames, comments):
    # Initialize an empty string to accumulate LaTeX code
    latex_content = ''

    # Generate LaTeX code for each filename in the list
    for i, filename in enumerate(filenames):
        if os.path.exists(filename):
            slide_number = os.path.splitext(os.path.basename(filename))[0]
            slide_comment = comments[i] if i < len(comments) else ''
            latex_content += rf'''
    \begin{{frame}}[fragile]{{{title} - Slide {slide_number}}}
        \frametitle{{{title} - Slide {slide_number}}}
        \begin{{columns}}
            \begin{{column}}{{0.6\textwidth}}
                \includegraphics[width=\textwidth]{{{filename}}}
            \end{{column}}
            \begin{{column}}{{0.4\textwidth}}
                \raggedleft
                \small
                {slide_comment}
            \end{{column}}
        \end{{columns}}
    \end{{frame}}
    '''

    return latex_content

# Example usage:
# List of filenames and comments for each slide set
dir_list = ["C:\\Users\\SVONDEHN\\wps-lx-compare\\strike_data"]
slide_list = [[] for i in range(len(dir_list))]

for i in range(len(dir_list)):
    files = os.listdir(dir_list[i])
    file_list = []
    for n in range(len(files)):
        if files[n].split('.')[-1] == 'png':
            file_list.append(f'{dir_list[i]}/{files[n]}')
        else:
            continue;
    slide_list[i] = [f for f in file_list]

print(slide_list)
slide_set1 = slide_list[0]
print(slide_set1)
comments1 = ['']
# slide_set2 = slide_list[1]
# slide_set3 = slide_list[2]
# slide_set4 = slide_list[3]
# slide_set5 = slide_list[4]
# slide_set6 = slide_list[5]
# slide_set7 = slide_list[6]
# slide_set8 = slide_list[7]
# LaTeX preamble and end code
latex_preamble = r'''
\documentclass[aspectratio=169]{beamer}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{array}
\title{Lightning detector comparison}
\author{Sterling von Dehn and Ash Richardson}
\institute{B.C. Wildfire Service}
\date{\today}

\begin{document}

\begin{frame}
	\titlepage
\end{frame}


'''

latex_end = r'''
\end{document}
'''

# Make the presentation

with open('presentation.tex', 'w') as file:
    file.write((latex_preamble +
                generate_slide_frames('Strike distance from LCF', slide_set1,comments1)
                + latex_end
                ).replace('_','\\_')) 
    
#os.system('pdflatex presentation.tex; rm *.log *.nav *.aux *.snm *.vrb; open presentation.pdf')

