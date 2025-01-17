import ConfigParser
import socket

# Use this only to create config file
def setDBLocation():
    hostname = socket.gethostname()
    config = ConfigParser.SafeConfigParser()

    optionName = 'voc2011dir_' + hostname

    try:
        config.read('config.cfg')
        if (not (config.has_option('Databases', optionName))):
            raise ConfigParser.NoOptionError('Invalid config file')
    except Exception as e:
        print('Config file not found or corrupted; creating new one.')
        config = ConfigParser.SafeConfigParser()
        config.add_section('Databases')

    if (hostname == 'soma'):
        dbLoc = '/home/vision/msmith/localDrive/msmith/ecse608project/VOC2011Train'
    elif (hostname == 'minecraft'):
        dbLoc = '/usr/local/data/ajesson/Projects/PycharmProjects/ecse608project/data/VOC2011Train/'

    config.set('Databases', optionName, dbLoc)

    with open('config.cfg', 'wb') as configfile:
        config.write(configfile)

    configfile.close()
