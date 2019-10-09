import dominate
from dominate.tags import meta, h3, table, tr, td, br, p, img, a
import os


class HTML:
    def __init__(self, web_dir, title, refresh=0):
        """

        :param web_dir: a directory that stores the webpage
        :param title: the webpage name
        :param refresh: how often the website refresh itself
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, text):
        with self.doc:
            h3(text)

    def add_images(self, img_pths, images_name, hyper_link, image_size=400):
        """
        add images to html file
        :param img_pths: (list)
        :param images_name: (list) 
        :param hyper_link:  (list)
        :param image_size: 
        :return: 
        """
        t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(t)
        with t:
            with tr():
                for img_pth, img_name, link in zip(img_pths, images_name, hyper_link):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:{}px" .format(image_size), src=os.path.join('images', img_pth))
                            br()
                            p(img_name)

    def save(self):
        """save the current content to the HMTL file"""
        html_file = '{}/index.html'.format(self.web_dir)
        print(html_file)
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':  # we show an example usage here.
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    image_paths, image_names, links = [], [], []
    for n in range(4):
        image_paths.append('image_%d.png' % n)
        image_names.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(image_paths, image_paths, links)
    html.save()
