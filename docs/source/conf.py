import mock

MOCK_MODULES = [
    'roboscheduler',
    'roboscheduler.cadence',
    'kaiju']
    'kaiju.robotGrid']

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()
