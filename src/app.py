import settings

from api import start_api


if __name__ == '__main__':
    start_api(settings.API_HOST, settings.API_PORT)