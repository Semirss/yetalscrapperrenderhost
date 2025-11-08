import mimetypes

def what(file, h=None):
    mime = mimetypes.guess_type(file)[0]
    if mime and mime.startswith('image/'):
        return mime.split('/')[-1]
    return None
