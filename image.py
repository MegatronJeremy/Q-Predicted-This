import pygame


class Image:
    image_dict = {}  # type: ignore

    @staticmethod
    def get_image(path, size, transparent_color=None):
        if path in Image.image_dict:
            return Image.image_dict[path]
        Image.image_dict[path] = pygame.transform.scale(pygame.image.load(path).convert(), size)
        if transparent_color:
            Image.image_dict[path].set_colorkey(transparent_color)
        return Image.image_dict[path]
