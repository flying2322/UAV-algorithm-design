\ Model TSP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  27 x_0_1 + 20 x_0_2 + 29 x_0_3 + 25 x_0_4 + 15 x_0_5 + 25 x_0_6 + 26 x_0_7
   + 11 x_0_8 + 23 x_0_9 + 27 x_1_0 + 8 x_1_2 + 11 x_1_3 + 12 x_1_4
   + 14 x_1_5 + 48 x_1_6 + 7 x_1_7 + 23 x_1_8 + 24 x_1_9 + 27 x_1_10
   + 20 x_2_0 + 8 x_2_1 + 17 x_2_3 + 15 x_2_4 + 5 x_2_5 + 40 x_2_6
   + 13 x_2_7 + 14 x_2_8 + 17 x_2_9 + 20 x_2_10 + 29 x_3_0 + 11 x_3_1
   + 17 x_3_2 + 3 x_3_4 + 20 x_3_5 + 53 x_3_6 + 4 x_3_7 + 29 x_3_8
   + 34 x_3_9 + 29 x_3_10 + 25 x_4_0 + 12 x_4_1 + 15 x_4_2 + 3 x_4_3
   + 18 x_4_5 + 50 x_4_6 + 4 x_4_7 + 26 x_4_8 + 32 x_4_9 + 25 x_4_10
   + 15 x_5_0 + 14 x_5_1 + 5 x_5_2 + 20 x_5_3 + 18 x_5_4 + 34 x_5_6
   + 16 x_5_7 + 9 x_5_8 + 14 x_5_9 + 15 x_5_10 + 25 x_6_0 + 48 x_6_1
   + 40 x_6_2 + 53 x_6_3 + 50 x_6_4 + 34 x_6_5 + 50 x_6_7 + 25 x_6_8
   + 30 x_6_9 + 25 x_6_10 + 26 x_7_0 + 7 x_7_1 + 13 x_7_2 + 4 x_7_3
   + 4 x_7_4 + 16 x_7_5 + 50 x_7_6 + 26 x_7_8 + 30 x_7_9 + 26 x_7_10
   + 11 x_8_0 + 23 x_8_1 + 14 x_8_2 + 29 x_8_3 + 26 x_8_4 + 9 x_8_5
   + 25 x_8_6 + 26 x_8_7 + 11 x_8_9 + 11 x_8_10 + 23 x_9_0 + 24 x_9_1
   + 17 x_9_2 + 34 x_9_3 + 32 x_9_4 + 14 x_9_5 + 30 x_9_6 + 30 x_9_7
   + 11 x_9_8 + 23 x_9_10 + 0 x_10_0 + 27 x_10_1 + 20 x_10_2 + 29 x_10_3
   + 25 x_10_4 + 15 x_10_5 + 25 x_10_6 + 26 x_10_7 + 11 x_10_8 + 23 x_10_9
Subject To
 visit_1: x_0_1 + x_2_1 + x_3_1 + x_4_1 + x_5_1 + x_6_1 + x_7_1 + x_8_1
   + x_9_1 = 1
 visit_2: x_0_2 + x_1_2 + x_3_2 + x_4_2 + x_5_2 + x_6_2 + x_7_2 + x_8_2
   + x_9_2 = 1
 visit_3: x_0_3 + x_1_3 + x_2_3 + x_4_3 + x_5_3 + x_6_3 + x_7_3 + x_8_3
   + x_9_3 = 1
 visit_4: x_0_4 + x_1_4 + x_2_4 + x_3_4 + x_5_4 + x_6_4 + x_7_4 + x_8_4
   + x_9_4 = 1
 visit_5: x_0_5 + x_1_5 + x_2_5 + x_3_5 + x_4_5 + x_6_5 + x_7_5 + x_8_5
   + x_9_5 = 1
 visit_6: x_0_6 + x_1_6 + x_2_6 + x_3_6 + x_4_6 + x_5_6 + x_7_6 + x_8_6
   + x_9_6 = 1
 visit_7: x_0_7 + x_1_7 + x_2_7 + x_3_7 + x_4_7 + x_5_7 + x_6_7 + x_8_7
   + x_9_7 = 1
 visit_8: x_0_8 + x_1_8 + x_2_8 + x_3_8 + x_4_8 + x_5_8 + x_6_8 + x_7_8
   + x_9_8 = 1
 visit_9: x_0_9 + x_1_9 + x_2_9 + x_3_9 + x_4_9 + x_5_9 + x_6_9 + x_7_9
   + x_8_9 = 1
 visit_10: x_0_10 + x_1_10 + x_2_10 + x_3_10 + x_4_10 + x_5_10 + x_6_10
   + x_7_10 + x_8_10 + x_9_10 = 1
 visit_10: x_0_1 + x_0_2 + x_0_3 + x_0_4 + x_0_5 + x_0_6 + x_0_7 + x_0_8
   + x_0_9 + x_0_10 = 1
 visit_10: x_1_2 + x_1_3 + x_1_4 + x_1_5 + x_1_6 + x_1_7 + x_1_8 + x_1_9
   + x_1_10 = 1
 visit_10: x_2_1 + x_2_3 + x_2_4 + x_2_5 + x_2_6 + x_2_7 + x_2_8 + x_2_9
   + x_2_10 = 1
 visit_10: x_3_1 + x_3_2 + x_3_4 + x_3_5 + x_3_6 + x_3_7 + x_3_8 + x_3_9
   + x_3_10 = 1
 visit_10: x_4_1 + x_4_2 + x_4_3 + x_4_5 + x_4_6 + x_4_7 + x_4_8 + x_4_9
   + x_4_10 = 1
 visit_10: x_5_1 + x_5_2 + x_5_3 + x_5_4 + x_5_6 + x_5_7 + x_5_8 + x_5_9
   + x_5_10 = 1
 visit_10: x_6_1 + x_6_2 + x_6_3 + x_6_4 + x_6_5 + x_6_7 + x_6_8 + x_6_9
   + x_6_10 = 1
 visit_10: x_7_1 + x_7_2 + x_7_3 + x_7_4 + x_7_5 + x_7_6 + x_7_8 + x_7_9
   + x_7_10 = 1
 visit_10: x_8_1 + x_8_2 + x_8_3 + x_8_4 + x_8_5 + x_8_6 + x_8_7 + x_8_9
   + x_8_10 = 1
 visit_10: x_9_1 + x_9_2 + x_9_3 + x_9_4 + x_9_5 + x_9_6 + x_9_7 + x_9_8
   + x_9_10 = 1
 R20: mu_0 + 100 x_0_1 - mu_1 <= 99
 R21: mu_0 + 100 x_0_2 - mu_2 <= 99
 R22: mu_0 + 100 x_0_3 - mu_3 <= 99
 R23: mu_0 + 100 x_0_4 - mu_4 <= 99
 R24: mu_0 + 100 x_0_5 - mu_5 <= 99
 R25: mu_0 + 100 x_0_6 - mu_6 <= 99
 R26: mu_0 + 100 x_0_7 - mu_7 <= 99
 R27: mu_0 + 100 x_0_8 - mu_8 <= 99
 R28: mu_0 + 100 x_0_9 - mu_9 <= 99
 R29: mu_0 + 100 x_0_10 - mu_10 <= 99
 R30: mu_1 + 100 x_1_2 - mu_2 <= 99
 R31: mu_1 + 100 x_1_3 - mu_3 <= 99
 R32: mu_1 + 100 x_1_4 - mu_4 <= 99
 R33: mu_1 + 100 x_1_5 - mu_5 <= 99
 R34: mu_1 + 100 x_1_6 - mu_6 <= 99
 R35: mu_1 + 100 x_1_7 - mu_7 <= 99
 R36: mu_1 + 100 x_1_8 - mu_8 <= 99
 R37: mu_1 + 100 x_1_9 - mu_9 <= 99
 R38: mu_1 + 100 x_1_10 - mu_10 <= 99
 R39: - mu_1 + mu_2 + 100 x_2_1 <= 99
 R40: mu_2 + 100 x_2_3 - mu_3 <= 99
 R41: mu_2 + 100 x_2_4 - mu_4 <= 99
 R42: mu_2 + 100 x_2_5 - mu_5 <= 99
 R43: mu_2 + 100 x_2_6 - mu_6 <= 99
 R44: mu_2 + 100 x_2_7 - mu_7 <= 99
 R45: mu_2 + 100 x_2_8 - mu_8 <= 99
 R46: mu_2 + 100 x_2_9 - mu_9 <= 99
 R47: mu_2 + 100 x_2_10 - mu_10 <= 99
 R48: - mu_1 + mu_3 + 100 x_3_1 <= 99
 R49: - mu_2 + mu_3 + 100 x_3_2 <= 99
 R50: mu_3 + 100 x_3_4 - mu_4 <= 99
 R51: mu_3 + 100 x_3_5 - mu_5 <= 99
 R52: mu_3 + 100 x_3_6 - mu_6 <= 99
 R53: mu_3 + 100 x_3_7 - mu_7 <= 99
 R54: mu_3 + 100 x_3_8 - mu_8 <= 99
 R55: mu_3 + 100 x_3_9 - mu_9 <= 99
 R56: mu_3 + 100 x_3_10 - mu_10 <= 99
 R57: - mu_1 + mu_4 + 100 x_4_1 <= 99
 R58: - mu_2 + mu_4 + 100 x_4_2 <= 99
 R59: - mu_3 + mu_4 + 100 x_4_3 <= 99
 R60: mu_4 + 100 x_4_5 - mu_5 <= 99
 R61: mu_4 + 100 x_4_6 - mu_6 <= 99
 R62: mu_4 + 100 x_4_7 - mu_7 <= 99
 R63: mu_4 + 100 x_4_8 - mu_8 <= 99
 R64: mu_4 + 100 x_4_9 - mu_9 <= 99
 R65: mu_4 + 100 x_4_10 - mu_10 <= 99
 R66: - mu_1 + mu_5 + 100 x_5_1 <= 99
 R67: - mu_2 + mu_5 + 100 x_5_2 <= 99
 R68: - mu_3 + mu_5 + 100 x_5_3 <= 99
 R69: - mu_4 + mu_5 + 100 x_5_4 <= 99
 R70: mu_5 + 100 x_5_6 - mu_6 <= 99
 R71: mu_5 + 100 x_5_7 - mu_7 <= 99
 R72: mu_5 + 100 x_5_8 - mu_8 <= 99
 R73: mu_5 + 100 x_5_9 - mu_9 <= 99
 R74: mu_5 + 100 x_5_10 - mu_10 <= 99
 R75: - mu_1 + mu_6 + 100 x_6_1 <= 99
 R76: - mu_2 + mu_6 + 100 x_6_2 <= 99
 R77: - mu_3 + mu_6 + 100 x_6_3 <= 99
 R78: - mu_4 + mu_6 + 100 x_6_4 <= 99
 R79: - mu_5 + mu_6 + 100 x_6_5 <= 99
 R80: mu_6 + 100 x_6_7 - mu_7 <= 99
 R81: mu_6 + 100 x_6_8 - mu_8 <= 99
 R82: mu_6 + 100 x_6_9 - mu_9 <= 99
 R83: mu_6 + 100 x_6_10 - mu_10 <= 99
 R84: - mu_1 + mu_7 + 100 x_7_1 <= 99
 R85: - mu_2 + mu_7 + 100 x_7_2 <= 99
 R86: - mu_3 + mu_7 + 100 x_7_3 <= 99
 R87: - mu_4 + mu_7 + 100 x_7_4 <= 99
 R88: - mu_5 + mu_7 + 100 x_7_5 <= 99
 R89: - mu_6 + mu_7 + 100 x_7_6 <= 99
 R90: mu_7 + 100 x_7_8 - mu_8 <= 99
 R91: mu_7 + 100 x_7_9 - mu_9 <= 99
 R92: mu_7 + 100 x_7_10 - mu_10 <= 99
 R93: - mu_1 + mu_8 + 100 x_8_1 <= 99
 R94: - mu_2 + mu_8 + 100 x_8_2 <= 99
 R95: - mu_3 + mu_8 + 100 x_8_3 <= 99
 R96: - mu_4 + mu_8 + 100 x_8_4 <= 99
 R97: - mu_5 + mu_8 + 100 x_8_5 <= 99
 R98: - mu_6 + mu_8 + 100 x_8_6 <= 99
 R99: - mu_7 + mu_8 + 100 x_8_7 <= 99
 R100: mu_8 + 100 x_8_9 - mu_9 <= 99
 R101: mu_8 + 100 x_8_10 - mu_10 <= 99
 R102: - mu_1 + mu_9 + 100 x_9_1 <= 99
 R103: - mu_2 + mu_9 + 100 x_9_2 <= 99
 R104: - mu_3 + mu_9 + 100 x_9_3 <= 99
 R105: - mu_4 + mu_9 + 100 x_9_4 <= 99
 R106: - mu_5 + mu_9 + 100 x_9_5 <= 99
 R107: - mu_6 + mu_9 + 100 x_9_6 <= 99
 R108: - mu_7 + mu_9 + 100 x_9_7 <= 99
 R109: - mu_8 + mu_9 + 100 x_9_8 <= 99
 R110: mu_9 + 100 x_9_10 - mu_10 <= 99
Bounds
 mu_0 <= 100
 mu_1 <= 100
 mu_2 <= 100
 mu_3 <= 100
 mu_4 <= 100
 mu_5 <= 100
 mu_6 <= 100
 mu_7 <= 100
 mu_8 <= 100
 mu_9 <= 100
 mu_10 <= 100
Binaries
 x_0_1 x_0_2 x_0_3 x_0_4 x_0_5 x_0_6 x_0_7 x_0_8 x_0_9 x_0_10 x_1_0 x_1_2
 x_1_3 x_1_4 x_1_5 x_1_6 x_1_7 x_1_8 x_1_9 x_1_10 x_2_0 x_2_1 x_2_3 x_2_4
 x_2_5 x_2_6 x_2_7 x_2_8 x_2_9 x_2_10 x_3_0 x_3_1 x_3_2 x_3_4 x_3_5 x_3_6
 x_3_7 x_3_8 x_3_9 x_3_10 x_4_0 x_4_1 x_4_2 x_4_3 x_4_5 x_4_6 x_4_7 x_4_8
 x_4_9 x_4_10 x_5_0 x_5_1 x_5_2 x_5_3 x_5_4 x_5_6 x_5_7 x_5_8 x_5_9 x_5_10
 x_6_0 x_6_1 x_6_2 x_6_3 x_6_4 x_6_5 x_6_7 x_6_8 x_6_9 x_6_10 x_7_0 x_7_1
 x_7_2 x_7_3 x_7_4 x_7_5 x_7_6 x_7_8 x_7_9 x_7_10 x_8_0 x_8_1 x_8_2 x_8_3
 x_8_4 x_8_5 x_8_6 x_8_7 x_8_9 x_8_10 x_9_0 x_9_1 x_9_2 x_9_3 x_9_4 x_9_5
 x_9_6 x_9_7 x_9_8 x_9_10 x_10_0 x_10_1 x_10_2 x_10_3 x_10_4 x_10_5 x_10_6
 x_10_7 x_10_8 x_10_9
End
