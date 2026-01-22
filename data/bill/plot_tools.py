'''
plot_tools.py

A collection of plot tools, can be useful
'''


import matplotlib.pyplot as plt


def plot(
        X,
        *,
        title = None,
        figsize = (10, 8)
):
        
        '''
        Just a simple plot. I didn't even need to write.

        Helpful for linux command python3 file.py ...

        Parameters
        ----------
        X: data to plot
        

        Returns
        -------
        A plot of the data:)
        '''

        plt.figure(figsize=figsize) 
        plt.imshow(X)
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.show()




def plot_multiple(
        X_list: list,
        *,
        title_list: list,
        max_per_row = 4,
        figsize = (15, 6),
        axis_off = True
):
        '''
        Plot multiple images

        Parameters
        ----------
        X_list:      a list of data to plot sequentially

        title_list:  a list of data to plot sequentially (no need to match size of X_list)

        max_per_row: maximum subfigs per row

        figsize:     size of figure
        
        axis_off:    'on' if you want the axis (this code was built for axis off, so there might be bugs)

        
        Returns
        -------
        A figure of multiple subplots
        '''

        import math
        import numpy as np

        # -------------------------
        # Validation
        # -------------------------
        if not X_list:
                raise ValueError("X_list is empty")

        if max_per_row <= 0:
                raise ValueError("max_per_row must be > 0")

        n_imgs = len(X_list)
        nrow = math.ceil(n_imgs / max_per_row)

        fig, axes = plt.subplots(
                nrow, max_per_row, figsize=figsize
        )

        # Make axes always 2D
        axes = np.atleast_2d(axes)

        for i in range(n_imgs):
                r = i // max_per_row
                c = i % max_per_row

                axes[r, c].imshow(X_list[i])

                if title_list is not None and i < len(title_list):
                    axes[r, c].set_title(title_list[i])

                axes[r, c].axis("off" if axis_off else "on")

        # Hide unused axes
        for i in range(n_imgs, nrow * max_per_row):
                r = i // max_per_row
                c = i % max_per_row
                axes[r, c].axis("off")

        plt.tight_layout()
        plt.show()





        

