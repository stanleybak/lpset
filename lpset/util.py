'''
Utilities for testing

Stanley Bak, 2018
'''

class Freezable():
    'a class where you can freeze the fields (prevent new fields from being created)'

    _frozen = False

    def freeze_attrs(self):
        'prevents any new attributes from being created in the object'
        self._frozen = True

    def __setattr__(self, key, value):
        if self._frozen and not hasattr(self, key):
            raise TypeError("{} does not contain attribute '{}' (object was frozen)".format(self, key))

        object.__setattr__(self, key, value)

def pair_almost_in(pair, pair_list, tol=1e-9):
    'check if a pair is in a pair list (up to small tolerance)'

    rv = False

    for a, b in pair_list:
        if abs(a - pair[0]) < tol and abs(b - pair[1]) < tol:
            rv = True
            break

    return rv

def assert_verts_equals(verts, check_list, tol=1e-5):
    '''check that the two lists of vertices are the same'''

    for v in check_list:
        assert pair_almost_in(v, verts, tol), "{} was not found in verts: {}".format(v, verts)

    for v in verts:
        assert pair_almost_in(v, check_list, tol), "verts contains {}, which was not in check_list: {}".format(
            v, check_list)

def assert_verts_is_box(verts, box, tol=1e-5):
    '''check that a list of verts is almost equal to the passed-in box using assertions

    box is [[xmin, xmax], [ymin, ymax]]
    '''

    pts = [(box[0][0], box[1][0]), (box[0][1], box[1][0]), (box[0][1], box[1][1]), (box[0][0], box[1][1])]

    assert_verts_equals(verts, pts, tol)
