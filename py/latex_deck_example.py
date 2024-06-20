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
slide_set1 = ['slide1.png', 'slide2.png', 'slide3.png']
slide_set2 = ['slide4.png', 'slide5.png']
comments_set1 = [
    'Comment for slide 1',
    'Comment for slide 2',
    'Comment for slide 3'
]
comments_set2 = [
    'Comment for slide 4',
    'Comment for slide 5'
]

# LaTeX preamble and end code
latex_preamble = r'''
\documentclass{beamer}
\usepackage{graphicx}
\usepackage{array}
\begin{document}
'''

latex_end = r'''
\end{document}
'''

# Make the presentation

with open('presentation.tex', 'w') as file:
    file.write(latex_preamble +
               generate_slide_frames('Slide Set 1', slide_set1, comments_set1) +
               generate_slide_frames('Slide Set 2', slide_set2, comments_set2) + 
               latex_content_set2 + latex_end) 


