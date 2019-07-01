import numpy as np

from nmrtools.math import normalize_spectrum
from nmrtools.partial import (AB, AB2, ABX, ABX3, AAXX, AABB)


def test_convert_refspec():
    refspec = [(1, 1), (2, 3), (3, 3), (4, 1)]
    new_refspec = normalize_spectrum(refspec, 2)
    print(sum([y for x, y in new_refspec]))
    assert new_refspec == [(1, 0.25), (2, 0.75), (3, 0.75), (4, 0.25)]


def test_AB():
    from .windnmr_defaults import ABdict
    refspec = [(134.39531364385073, 0.3753049524455757),
               (146.39531364385073, 1.6246950475544244),
               (153.60468635614927, 1.6246950475544244),
               (165.60468635614927, 0.3753049524455757)]

    refspec_normalized = normalize_spectrum(refspec, 2)
    print('ref normalized:')
    print(refspec_normalized)
    print(sum([y for x, y in refspec_normalized]))
    print(sum(refspec_normalized[1]))
    testspec = AB(**ABdict)
    print('testspec:')
    print(testspec)
    print(sum([y for x, y in testspec]))
    np.testing.assert_array_almost_equal(testspec,
                                         refspec_normalized,
                                         decimal=2)


def test_AB2():
    from .windnmr_defaults import dcp
    refspec = [(-8.892448165479056, 0.5434685012269458),
               (-2.300397938882746, 0.7780710767178313),
               (0.0, 1),
               (6.59205022659631, 1.6798068052995172),
               (22.865501607924635, 2.6784604220552235),
               (23.542448165479055, 2.4565314987730544),
               (30.134498392075365, 1.5421221179826525),
               (31.75794977340369, 1.3201931947004837),
               (55.300397938882746, 0.001346383244293953)]
    refspec = normalize_spectrum(refspec, 3)
    print('ref normalized;')
    print(refspec)
    print(sum([y for x, y in refspec]))
    testspec = sorted(AB2(**dcp))
    # testspec.sort()
    print('testspec:')
    print(testspec)
    print(sum([y for x, y in testspec]))
    np.testing.assert_array_almost_equal(sorted(testspec), refspec, decimal=2)


def test_ABX():
    from .windnmr_defaults import ABXdict
    refspec = sorted([(-9.48528137423857, 0.2928932188134524),
                      (-6.816653826391969, 0.44529980377477096),
                      (2.5147186257614305, 1.7071067811865475),
                      (5.183346173608031, 1.554700196225229),
                      (7.4852813742385695, 1.7071067811865475),
                      (14.816653826391969, 1.554700196225229),
                      (19.485281374238568, 0.2928932188134524),
                      (26.81665382639197, 0.44529980377477096),
                      (95.0, 1),
                      (102.3313724521534, 0.9902903378454601),
                      (97.6686275478466, 0.9902903378454601),
                      (105.0, 1),
                      (80.69806479936946, 0.009709662154539944),
                      (119.30193520063054, 0.009709662154539944)])
    refspec = normalize_spectrum(refspec, 3)
    print('ref normalized;')
    print(refspec)
    print(sum([y for x, y in refspec]))
    testspec = sorted(ABX(**ABXdict))
    print('testspec:')
    print(testspec)
    print(sum([y for x, y in testspec]))
    np.testing.assert_array_almost_equal(testspec, refspec, decimal=2)


def test_ABX3():
    from .windnmr_defaults import ABX3dict
    refspec = (
        [(124.2804555427071, 0.04365107831800394),
         (131.2804555427071, 0.13095323495401182),
         (136.2804555427071, 0.20634892168199606),
         (138.2804555427071, 0.13095323495401182),
         (142.7195444572929, 0.20634892168199606),
         (143.2804555427071, 0.6190467650459882),
         (145.2804555427071, 0.04365107831800394),
         (149.7195444572929, 0.6190467650459882),
         (150.2804555427071, 0.6190467650459882),
         (154.7195444572929, 0.04365107831800394),
         (156.7195444572929, 0.6190467650459882),
         (157.2804555427071, 0.20634892168199606),
         (161.7195444572929, 0.13095323495401182),
         (163.7195444572929, 0.20634892168199606),
         (168.7195444572929, 0.13095323495401182),
         (175.7195444572929, 0.04365107831800394)]
    )

    testspec = sorted(ABX3(**ABX3dict, normalize=True))
    # refspec appropriate if normalize=False, but want to cover all code, so
    sum_intensities = sum([y for x, y in refspec])
    print('target total intensity is: ', sum_intensities)  # 4
    testspec = normalize_spectrum(testspec, sum_intensities)
    np.testing.assert_array_almost_equal(testspec, refspec, decimal=2)


def test_AAXX():
    from .windnmr_defaults import AAXXdict
    refspec = sorted(
        [(173.0, 2), (127.0, 2), (169.6828402774396, 0.4272530047525843),
         (164.6828402774396, 0.5727469952474157),
         (135.3171597225604, 0.5727469952474157),
         (130.3171597225604, 0.4272530047525843),
         (183.6009478460092, 0.20380477476124093),
         (158.6009478460092, 0.7961952252387591),
         (141.3990521539908, 0.7961952252387591),
         (116.39905215399081, 0.20380477476124093)]
    )
    testspec = sorted(AAXX(**AAXXdict, normalize=True))
    # refspec appropriate if normalize=False, but want to cover all code, so
    sum_intensities = sum([y for x, y in refspec])
    print('target total intensity is: ', sum_intensities)  # 8
    testspec = normalize_spectrum(testspec, sum_intensities)
    np.testing.assert_array_almost_equal(testspec, refspec, decimal=2)


def test_AABB():
    from .windnmr_defaults import AABBdict
    refspec = (
        [(92.22140228380421, 0.10166662880050205),
         (96.52049869174374, 0.49078895567299158),
         (98.198417381432606, 0.24246823712843479),
         (101.65157834035199, 0.60221210650067936),
         (106.27391102440463, 0.32609441972096737),
         (110.31814797792887, 0.50200815047635172),
         (132.76708635908275, 0.7056360644549744),
         (134.42329797582249, 1.8983333711994956),
         (140.8425800020548, 1.5528911909909029),
         (142.5204986917436, 3.5092110443270066),
         (144.57608469738022, 4.560944163972513),
         (147.47995633005294, 1.4979918495236497),
         (152.52004366994717, 1.4979918495236475),
         (155.42391530261983, 4.5609441639725166),
         (157.47950130825635, 3.5092110443270097),
         (159.1574199979452, 1.5528911909909029),
         (165.57670202417737, 1.8983333711994945),
         (167.2329136409173, 0.70563606445497351),
         (189.68185202207124, 0.50200815047635128),
         (193.72608897559536, 0.32609441972096725),
         (198.34842165964801, 0.6022121065006798),
         (201.80158261856747, 0.24246823712843413),
         (203.47950130825626, 0.49078895567299208),
         (207.77859771619566, 0.10166662880050205)]
    )
    result = AABB(**AABBdict, normalize=False)
    x, y = result.T
    testspec = sorted(list(zip(x, y)))
    assert np.allclose(testspec, refspec)
