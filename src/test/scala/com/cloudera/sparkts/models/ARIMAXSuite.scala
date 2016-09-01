/**
 * Copyright (c) 2016, Cloudera, Inc. All Rights Reserved.
 *
 * Cloudera, Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"). You may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for
 * the specific language governing permissions and limitations under the
 * License.
 */

package com.cloudera.sparkts.models

import breeze.linalg.{DenseMatrix, DenseVector => BreezeDenseVector}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.mllib.linalg.DenseVector
import org.scalatest.FunSuite
import org.scalatest.Matchers._

class ARIMAXSuite extends FunSuite {
  val xregFile = getClass.getClassLoader.getResourceAsStream("arimax_xreg.csv")
  val rawXreg = scala.io.Source.fromInputStream(xregFile).getLines().toArray.map(_.toDouble)
  val xreg = new DenseMatrix(rows = rawXreg.length, cols = 1, data = rawXreg)
  val tsFile = getClass.getClassLoader.getResourceAsStream("arimax_data.csv")
  val rawTs = scala.io.Source.fromInputStream(tsFile).getLines().toArray.map(_.toDouble)
  val ts = new DenseVector(rawTs)

  val exogenous = new DenseMatrix(rows = 116, cols = 4, data = Array(416,393,444,445,426,435,471,397,454,416,424,395,401,471,400,418,476,436,442,472,492,443,418,417,423,382,433,409,436,437,372,419,423,415,432,413,361,415,437,391,395,468,415,386,410,437,401,446,492,443,438,417,384,418,403,408,380,422,432,405,437,444,485,426,411,440,400,440,432,439,431,384,404,439,401,401,427,375,411,428,376,407,403,454,478,418,428,401,467,456,446,509,406,431,458,469,450,462,538,435,485,439,451,457,495,479,418,423,430,477,423,462,481,406,450,405,
    0,0,0,0,1,1,0,0,0,0,0,1.0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,
    28,28,28,28,28,28,29,29,29,29,29,29,29,21,21,21,21,21,21,21,28,28,28,28,28,28,28,21,21,21,21,21,21,21,30,30,30,30,30,30,30,42,42,42,15,15,15,15,19,19,19,19,19,19,19,23,23,23,23,23,23,23,25,25,25,25,25,25,25,16,16,16,16,16,16,16,17,17,17,17,17,17,17,21,21,21,21,21,26,26,26,35,35,35,35,35,35,35,34,34,34,34,34,34,34,25,25,25,25,25,25,25,24,24,24,24,
    55,57,53,55,57,50,50,53,51,55,48,46,42,41,48,48,55,59,57,55,59,53,46,44,41,33,32,42,41,37,44,41,44,42,41,37,46,46,37,44,42,39,41,35,57,62,55,53,53,55,55,42,46,42,42,48,50,44,50,48,50,57,55,59,59,53,57,60,55,51,44,42,41,48,50,46,41,39,50,53,48,42,39,33,44,37,35,41,54,53,50,47,52,52,57,53,53,50,55,46,51,56,57,57,57,53,50,42,49,52,53,50,46,48,49,52))
  val timeSeries = new DenseVector(Array(93.0,82,109,110,109,84,100,91,119,78,99,92,76,99,84,103,107,106,106,89,121,103,92,94,99,94,90,99,100,125,78,95,92,84,99,88,85,121,119,94,89,121,110,110,78,88,86,77,106,127,91,98,108,110,88,118,112,104,97,100,97,96,95,111,84,102,98,110,108,92,121,104,109,105,93,74,106,118,97,109,90,91,95,95,111,112,96,122,108,96,78,124,79,89,98,127,110,92,120,109,106,124,135,110,98,108,109,103,106,92,89,82,118,94,112,86))
  val tsTest = new BreezeDenseVector(Array(100.0 ,98 ,102 ,98 ,112 ,99 ,99 ,87 ,103 ,115 ,101 ,125 ,117 ,109 ,111 ,105))
  val xregTest = new DenseMatrix(rows = 16, cols = 4, data = Array( 465,453,472,454,432,431,475,393,437,537,462,539,471,455,466,490,
    1,1,0,0,0,0,0,1,1,0,0,0,0,0,1.0,1,
    24,24,25,25,25,25,25,25,25,23,23,23,23,23,23,23,
    51,54,49,46,42,41,45,46,48,41,42,48,43,47,48,46 ))

  /**
    * 32 tests using different combinations of parameters.
    */
  test("1 ARIMAX(1,1,1) 0 t t"){
    // c, ar, ma, 4xreg
    val model1 = ARIMAX.fitModel(1, 1, 1, timeSeries, exogenous, 0, true, true)
    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
  }
  test("2 ARIMAX(1,1,1) 0 t f"){
    // c, ar, ma, 4xreg
    val model2 = ARIMAX.fitModel(1, 1, 1, timeSeries, exogenous, 0, true, false)
    assert( model2.coefficients.length == 7)
    val results = model2.predict(tsTest, xregTest)
  }
  test("3 ARIMAX(1,1,1) 0 f f"){
    // c, ar, ma
    val model3 = ARIMAX.fitModel(1, 1, 1, timeSeries, exogenous, 0, false, false)
    assert( model3.coefficients.length == 3)
    val results = model3.predict(tsTest, xregTest)
  }
  test("4 ARIMAX(1,1,1) 0 f t"){
    // c, ar, ma
    val model4 = ARIMAX.fitModel(1, 1, 1, timeSeries, exogenous, 0, false, true)
    assert( model4.coefficients.length == 3)
    val results = model4.predict(tsTest, xregTest)
  }
  test("5 ARIMAX(1,1,1) 1 t t"){
    // c, ar, ma, 4xreg x 2
    val model5 = ARIMAX.fitModel(1, 1, 1, timeSeries, exogenous, 1, true, true)
    assert( model5.coefficients.length == 11)
    val results = model5.predict(tsTest, xregTest)
  }
  test("6 ARIMAX(1,1,1) 1 t f"){
    // c, ar, ma, 4xreg x 2
    val model6 = ARIMAX.fitModel(1, 1, 1, timeSeries, exogenous, 1, true, false)
    assert( model6.coefficients.length == 11)
    val results = model6.predict(tsTest, xregTest)
  }
  test("7 ARIMAX(1,1,1) 1 f f"){
    // c, ar, ma, 4xreg
    val model7 = ARIMAX.fitModel(1, 1, 1, timeSeries, exogenous, 1, false, false)
    assert( model7.coefficients.length == 7)
    val results = model7.predict(tsTest, xregTest)
  }
  test("8 ARIMAX(1,1,1) 1 f t"){
    // c, ar, ma, 4xreg
    val model8 = ARIMAX.fitModel(1, 1, 1, timeSeries, exogenous, 1, false, true)
    assert( model8.coefficients.length == 7)
    val results = model8.predict(tsTest, xregTest)
  }
  val p = 2
  test(s"9 ARIMAX(2,1,1) 0 t t"){
    // c, ar, ma, 4xreg
    val model1 = ARIMAX.fitModel(p, 1, 1, timeSeries, exogenous, 0, true, true)
    assert( model1.coefficients.length == 8)
    val results = model1.predict(tsTest, xregTest)
  }
  test(s"10 ARIMAX(2,1,1) 0 t f"){
    // c, ar, ma, 4xreg
    val model2 = ARIMAX.fitModel(p, 1, 1, timeSeries, exogenous, 0, true, false)
    assert( model2.coefficients.length == 8)
    val results = model2.predict(tsTest, xregTest)
  }
  test(s"11 ARIMAX(2,1,1) 0 f f"){
    // c, ar, ma
    val model3 = ARIMAX.fitModel(p, 1, 1, timeSeries, exogenous, 0, false, false)
    assert( model3.coefficients.length == 4)
    val results = model3.predict(tsTest, xregTest)
  }
  test(s"12 ARIMAX(2,1,1) 0 f t"){
    // c, ar, ma
    val model4 = ARIMAX.fitModel(p, 1, 1, timeSeries, exogenous, 0, false, true)
    assert( model4.coefficients.length == 4)
    val results = model4.predict(tsTest, xregTest)
  }
  test(s"13 ARIMAX(2,1,1) 1 t t"){
    // c, ar, ma, 4xreg x 2
    val model5 = ARIMAX.fitModel(p, 1, 1, timeSeries, exogenous, 1, true, true)
    assert( model5.coefficients.length == 12)
    val results = model5.predict(tsTest, xregTest)
  }
  test(s"14 ARIMAX(2,1,1) 1 t f"){
    // c, ar, ma, 4xreg x 2
    val model6 = ARIMAX.fitModel(p, 1, 1, timeSeries, exogenous, 1, true, false)
    assert( model6.coefficients.length == 12)
    val results = model6.predict(tsTest, xregTest)
  }
  test(s"15 ARIMAX(2,1,1) 1 f f"){
    // c, ar, ma, 4xreg
    val model7 = ARIMAX.fitModel(p, 1, 1, timeSeries, exogenous, 1, false, false)
    assert( model7.coefficients.length == 8)
    val results = model7.predict(tsTest, xregTest)
  }
  test(s"16 ARIMAX(2,1,1) 1 f t"){
    // c, ar, ma, 4xreg
    val model8 = ARIMAX.fitModel(p, 1, 1, timeSeries, exogenous, 1, false, true)
    assert( model8.coefficients.length == 8)
    val results = model8.predict(tsTest, xregTest)
  }
  val q = 2
  test(s"17 ARIMAX(2,1,2) 0 t t"){
    // c, ar, ma, 4xreg
    val model1 = ARIMAX.fitModel(p, 1, q, timeSeries, exogenous, 0, true, true)
    assert( model1.coefficients.length == 9)
    val results = model1.predict(tsTest, xregTest)
  }
  test(s"18 ARIMAX(2,1,2) 0 t f"){
    // c, ar, ma, 4xreg
    val model2 = ARIMAX.fitModel(p, 1, q, timeSeries, exogenous, 0, true, false)
    assert( model2.coefficients.length == 9)
    val results = model2.predict(tsTest, xregTest)
  }
  test(s"19 ARIMAX(2,1,2) 0 f f"){
    // c, ar, ma
    val model3 = ARIMAX.fitModel(p, 1, q, timeSeries, exogenous, 0, false, false)
    assert( model3.coefficients.length == 5)
    val results = model3.predict(tsTest, xregTest)
  }
  test(s"20 ARIMAX(2,1,2) 0 f t"){
    // c, ar, ma
    val model4 = ARIMAX.fitModel(p, 1, q, timeSeries, exogenous, 0, false, true)
    assert( model4.coefficients.length == 5)
    val results = model4.predict(tsTest, xregTest)
  }
  test(s"21 ARIMAX(2,1,2) 1 t t"){
    // c, ar, ma, 4xreg x 2
    val model5 = ARIMAX.fitModel(p, 1, q, timeSeries, exogenous, 1, true, true)
    assert( model5.coefficients.length == 13)
    val results = model5.predict(tsTest, xregTest)
  }
  test(s"22 ARIMAX(2,1,2) 1 t f"){
    // c, ar, ma, 4xreg x 2
    val model6 = ARIMAX.fitModel(p, 1, q, timeSeries, exogenous, 1, true, false)
    assert( model6.coefficients.length == 13)
    val results = model6.predict(tsTest, xregTest)
  }
  test(s"23 ARIMAX(2,1,2) 1 f f"){
    // c, ar, ma, 4xreg
    val model7 = ARIMAX.fitModel(p, 1, q, timeSeries, exogenous, 1, false, false)
    assert( model7.coefficients.length == 9)
    val results = model7.predict(tsTest, xregTest)
  }
  test(s"24 ARIMAX(2,1,2) 1 f t"){
    // c, ar, ma, 4xreg
    val model8 = ARIMAX.fitModel(p, 1, q, timeSeries, exogenous, 1, false, true)
    assert( model8.coefficients.length == 9)
    val results = model8.predict(tsTest, xregTest)
  }
  val d = 2
  test(s"25 ARIMAX(2,2,2) 0 t t"){
    // c, ar, ma, 4xreg
    val model1 = ARIMAX.fitModel(p, d, q, timeSeries, exogenous, 0, true, true)
    assert( model1.coefficients.length == 9)
    val results = model1.predict(tsTest, xregTest)
  }
  test(s"26 ARIMAX(2,2,2) 0 t f"){
    // c, ar, ma, 4xreg
    val model2 = ARIMAX.fitModel(p, d, q, timeSeries, exogenous, 0, true, false)
    assert( model2.coefficients.length == 9)
    val results = model2.predict(tsTest, xregTest)
  }
  test(s"27 ARIMAX(2,2,2) 0 f f"){
    // c, ar, ma
    val model3 = ARIMAX.fitModel(p, d, q, timeSeries, exogenous, 0, false, false)
    assert( model3.coefficients.length == 5)
    val results = model3.predict(tsTest, xregTest)
  }
  test(s"28 ARIMAX(2,2,2) 0 f t"){
    // c, ar, ma
    val model4 = ARIMAX.fitModel(p, d, q, timeSeries, exogenous, 0, false, true)
    assert( model4.coefficients.length == 5)
    val results = model4.predict(tsTest, xregTest)
  }
  test(s"29 ARIMAX(2,2,2) 1 t t"){
    // c, ar, ma, 4xreg x 2
    val model5 = ARIMAX.fitModel(p, d, q, timeSeries, exogenous, 1, true, true)
    assert( model5.coefficients.length == 13)
    val results = model5.predict(tsTest, xregTest)
  }
  test(s"30 ARIMAX(2,2,2) 1 t f"){
    // c, ar, ma, 4xreg x 2
    val model6 = ARIMAX.fitModel(p, d, q, timeSeries, exogenous, 1, true, false)
    assert( model6.coefficients.length == 13)
    val results = model6.predict(tsTest, xregTest)
  }
  test(s"31 ARIMAX(2,2,2) 1 f f"){
    // c, ar, ma, 4xreg
    val model7 = ARIMAX.fitModel(p, d, q, timeSeries, exogenous, 1, false, false)
    assert( model7.coefficients.length == 9)
    val results = model7.predict(tsTest, xregTest)
  }
  test(s"32 ARIMAX(2,2,2) 1 f t"){
    // c, ar, ma, 4xreg
    val model8 = ARIMAX.fitModel(p, d, q, timeSeries, exogenous, 1, false, true)
    assert( model8.coefficients.length == 9)
    val results = model8.predict(tsTest, xregTest)
  }

  test("ARIMAX(1,1,1) fitting model - comparision with R auto.arima method"){
/*  summary(auto.arima(data,xreg = xreg), trace=True)
    Series: data ARIMAX(1,1,1)
    Coefficients:
      ar1      ma1       V1
    0.7074  -0.9300  -2.1703
    s.e.  0.0634   0.0315   4.7266sigma^2 estimated as 0.2256:  log likelihood=-172.1
    AIC=352.19   AICc=352.35   BIC=366.39Training set error measures:
      ME      RMSE       MAE       MPE      MAPE     MASE        ACF1
    Training set 0.04912198 0.4712798 0.2478504 0.1107324 0.6106924 1.248967 -0.02801876*/
    val arR = 0.7074
    val maR = -0.9300
    val xcoeffR = -2.1703
    val model = ARIMAX.fitModel(1, 1, 1, ts, xreg, 0)
    val Array(c, ar, ma, xregCoeff) = model.coefficients
    ar should be (arR +- 0.3)
    ma should be (maR +- 0.5)

    println(s"AR spark-ts - R difference: ${"%1.3f".format(math.abs(((arR - ar)/arR)*100))} %")
    println(s"MA spark-ts - R difference: ${"%1.3f".format(math.abs(((maR - ma)/maR)*100))} %")
    println(s"Xreg coefficient difference: ${"%1.3f".format(math.abs(((xcoeffR - xregCoeff)/xcoeffR)*100))} %")
  }
  test("ARIMAX(1,1,1) with xMaxLag = 1"){
    val model = ARIMAX.fitModel(1, 1, 1, ts, xreg, 1)
    assert( model.coefficients.length == 5)
  }
  test("ARIMAX(2,1,1)"){
    val model = ARIMAX.fitModel(2, 1, 1, ts, xreg, 0)
    assert( model.coefficients.length == 5)
  }
  test("ARIMAX(2,1,2)"){
    val model = ARIMAX.fitModel(2, 1, 2, ts, xreg, 0)
    assert( model.coefficients.length == 6)
  }
  test("ARIMAX(1,1,3)"){
    val model = ARIMAX.fitModel(1, 1, 3, ts, xreg, 0)
    assert( model.coefficients.length == 6)
  }

  test("ARIMAX prediction using simple data and coefficients from R") {
    val model = new ARIMAXModel(1, 1, 1, 0, Array(0.003, 0.1659, -0.999, 0.3236))
    val exogenous = new DenseMatrix(rows = 6, cols = 1, data = Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    val timeseries = new BreezeDenseVector(Array(38.1, 38.2, 39.3, 39.4, 39.5, 39.6))
    val results = model.predict(timeseries, exogenous)

    results.map( value => value should be (39.0 +- 3))
  }

  test("Predict ARIMAX(1,1,1) using coefficients from R"){
    val arR = 0.7074
    val maR = -0.9300
    val xcoeffR1 = -2.1703
    val xcoeffR2 = 1.05
    val model = new ARIMAXModel(1, 1, 1, 1, Array(0.003, arR, maR, xcoeffR1, xcoeffR2))

    val timeseries = new BreezeDenseVector(ts.toArray)
    val results = model.predict(timeseries, xreg)

    results.map( value => value should be (41.0 +- 3))
  }

  test("Predict ARIMAX(1,1,1) using our coefficients"){
    val fittedModel = ARIMAX.fitModel(1, 1, 1, ts, xreg, 1, false, false)
    val timeseries = new BreezeDenseVector(ts.toArray)

    val results  = fittedModel.predict(timeseries, xreg)

    results.map( value => value should be (41.0 +- 3))
  }
  
  test("ARIMAX(1,1,1) train and test - 2x260") {
    val exogenous = new DenseMatrix(rows = 260, cols = 2, data = Array(1.1255,1.1229,1.1138,1.1138,1.1138,1.1146,1.1162,1.1139,1.1185,1.1268,1.1268,1.1268,1.1305,1.1320,1.1228,1.1312,1.1419,1.1419,1.1419,1.1250,1.1155,1.1150,1.1241,1.1151,1.1151,1.1151,1.1170,1.1204,1.1203,1.1153,1.1160,1.1160,1.1160,1.1236,1.1224,1.1266,1.1254,1.1362,1.1362,1.1362,1.1373,1.1374,1.1410,1.1439,1.1360,1.1360,1.1360,1.1333,1.1373,1.1354,1.1313,1.1084,1.1084,1.1084,1.1011,1.1061,1.1085,1.0930,1.1017,1.1017,1.1017,1.1032,1.0976,1.0935,1.0883,1.0864,1.0864,1.0864,1.0776,1.0711,1.0716,1.0726,1.0764,1.0764,1.0764,1.0723,1.0670,1.0666,1.0687,1.0688,1.0688,1.0688,1.0631,1.0651,1.0586,1.0612,1.0580,1.0580,1.0580,1.0579,1.0600,1.0612,1.0671,1.0902,1.0902,1.0902,1.0809,1.0875,1.0941,1.0943,1.0950,1.0950,1.0950,1.0983,1.0990,1.0933,1.0841,1.0836,1.0836,1.0836,1.0870,1.0952,1.0916,1.0947,1.0947,1.0947,1.0947,1.0962,1.0952,1.0926,1.0887,1.0887,1.0887,1.0887,1.0898,1.0746,1.0742,1.0868,1.0861,1.0861,1.0861,1.0888,1.0836,1.0816,1.0893,1.0914,1.0914,1.0914,1.0892,1.0868,1.0907,1.0893,1.0808,1.0808,1.0808,1.0815,1.0837,1.0888,1.0903,1.0920,1.0920,1.0920,1.0884,1.0919,1.0933,1.1206,1.1202,1.1202,1.1202,1.1101,1.1236,1.1257,1.1347,1.1275,1.1275,1.1275,1.1180,1.1166,1.1136,1.1084,1.1096,1.1096,1.1096,1.1026,1.1002,1.0981,1.1027,1.1006,1.1006,1.1006,1.0888,1.0872,1.0856,1.0901,1.0970,1.0970,1.0970,1.0953,1.1028,1.0973,1.0857,1.1090,1.1090,1.1090,1.1119,1.1109,1.1064,1.1311,1.1279,1.1279,1.1279,1.1271,1.1212,1.1171,1.1154,1.1154,1.1154,1.1154,1.1154,1.1194,1.1324,1.1385,1.1432,1.1432,1.1432,1.1380,1.1367,1.1336,1.1364,1.1363,1.1363,1.1363,1.1390,1.1396,1.1298,1.1252,1.1284,1.1284,1.1284,1.1306,1.1343,1.1379,1.1355,1.1263,1.1263,1.1263,1.1264,1.1287,1.1303,1.1358,1.1403,1.1403,1.1403,1.1493,1.1569,1.1505,1.1439,1.1427,1.1427,1.1427,1.1395,1.1375,1.1409,1.1389,1.1348,1.1348,1.1348,1.1324,1.132427,1.132454,
      46.25000,46.75000,46.05000,46.05000,46.05000,46.05000,45.94000,44.15000,45.92000,44.63000,44.63000,44.63000,44.00000,44.59000,47.15000,46.90000,44.68000,44.68000,44.68000,46.68000,45.83000,44.48000,44.91000,45.70000,45.70000,45.70000,44.43000,45.23000,45.09000,44.74000,45.54000,45.54000,45.54000,46.26000,48.53000,47.81000,49.43000,49.63000,49.63000,49.63000,47.10000,46.66000,46.64000,46.38000,47.26000,47.26000,47.26000,45.89000,45.55000,45.20000,45.38000,44.60000,44.60000,44.60000,43.98000,43.20000,45.94000,46.06000,46.59000,46.59000,46.59000,46.14000,47.90000,46.32000,45.20000,44.29000,44.29000,44.29000,43.87000,44.21000,42.93000,41.75000,40.74000,40.74000,40.74000,41.74000,40.67000,40.75000,40.54000,40.39000,40.39000,40.39000,41.75000,42.87000,43.04000,43.04000,41.71000,41.71000,41.71000,41.65000,41.85000,39.94000,41.08000,39.97000,39.97000,39.97000,37.65000,37.51000,37.16000,36.76000,35.62000,35.62000,35.62000,36.31000,37.35000,35.52000,34.95000,34.73000,34.73000,34.73000,34.74000,36.14000,37.50000,38.10000,38.10000,38.10000,38.10000,36.81000,37.87000,36.60000,37.04000,37.04000,37.04000,37.04000,36.76000,35.97000,33.97000,33.27000,33.16000,33.16000,33.16000,31.41000,30.44000,30.48000,31.20000,29.42000,29.42000,29.42000,29.42000,28.46000,26.55000,29.53000,32.19000,32.19000,32.19000,30.34000,31.45000,32.30000,33.22000,33.62000,33.62000,33.62000,31.62000,29.88000,32.28000,31.72000,30.89000,30.89000,30.89000,29.69000,27.94000,27.45000,26.21000,29.44000,29.44000,29.44000,29.44000,29.04000,30.66000,30.77000,29.64000,29.64000,29.64000,33.39000,31.87000,32.15000,33.07000,32.78000,32.78000,32.78000,33.75000,34.40000,34.66000,34.57000,35.92000,35.92000,35.92000,37.90000,36.50000,38.29000,37.84000,38.50000,38.50000,38.50000,37.18000,36.34000,38.46000,40.20000,39.44000,39.44000,39.44000,41.52000,41.45000,39.79000,39.46000,39.46000,39.46000,39.46000,39.39000,38.28000,38.32000,38.34000,36.79000,36.79000,36.79000,35.70000,35.89000,37.75000,37.26000,39.72000,39.72000,39.72000,40.36000,42.17000,41.76000,41.50000,40.36000,40.36000,40.36000,39.78000,41.08000,42.63000,43.18000,43.73000,43.73000,43.73000,42.64000,44.04000,45.33000,46.03000,45.92000,45.92000,45.92000,44.78000,43.65000,43.78000,44.32000,44.66000,44.66000,44.66000,43.44000,44.66000,46.23000,46.70000,46.21000,46.21000,46.21000,47.72000,47.72572,47.73144))
    val timeSeries = new DenseVector(Array(38.3333,38.3333,39.0833,39.0833,39.0833,39.0833,39.5000,39.5000,39.5000,39.5000,39.0833,39.0833,39.0833,38.6667,38.6667,38.6667,38.6667,38.6667,38.6667,38.6667,39.0833,39.0833,39.0833,39.4167,39.4167,39.4167,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.0000,39.4167,39.4167,39.4167,39.8333,41.4167,41.4167,41.4167,41.0000,40.4167,40.4167,39.0000,39.0000,38.9167,38.9167,38.9167,38.9167,38.9167,38.9167,39.4167,40.0000,40.0000,40.0000,38.5833,39.4167,40.0000,38.7500,38.7500,40.0000,38.5833,38.5833,39.4167,39.4167,39.4167,38.5833,40.3333,40.3333,40.3333,40.3333,40.3333,40.3333,40.3333,40.3333,40.3333,40.3333,38.5833,39.4167,40.2500,40.0000,40.0000,40.0000,39.4167,39.4167,38.2500,40.2500,40.8333,40.8333,40.8333,40.8333,40.8333,40.8333,40.3333,40.8333,40.8333,40.8333,40.8333,40.8333,40.8333,40.8333,40.8333,40.8333,40.8333,40.8333,40.8333,40.5833,40.5833,39.3333,39.3333,39.3333,40.5833,40.5833,40.5833,40.5833,40.5833,40.5833,40.5833,40.1667,40.5833,40.5833,42.0833,42.0833,42.0833,42.0833,42.5000,42.5000,41.6667,41.6667,41.6667,41.6667,41.2500,41.2500,41.2500,41.2500,41.2500,41.2500,41.2500,41.2500,41.2500,41.2500,41.2500,41.2500,41.2500,41.2500,41.2500,41.2500,41.2500,41.2500,42.0833,42.0833,42.0833,41.6667,41.6667,41.6667,41.6667,41.6667,41.6667,41.6667,41.6667,41.2500,41.2500,41.2500,41.2500,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,40.6667,41.0833,41.0833,41.0833,41.0833,41.0833,41.0833,40.7500,41.6667,40.8333,40.8333,40.8333,41.6667,41.6667,41.6667,40.5833,44.0833,44.0833,44.0833,44.0833,41.6667,41.0833,41.6667,41.6667,41.6667,41.1667,41.1667,41.1667,41.6667,41.6667,42.0833,42.0833,42.0833,41.3333,42.0833,42.1260))
    val modelTrain = ARIMAX.fitModel(1, 1, 1, timeSeries, exogenous, 1, false, false)

    val tsTest = new BreezeDenseVector(Array(41.0833, 41.0833, 40.7500, 41.6667, 40.8333, 40.8333, 40.8333, 41.6667, 41.6667, 41.6667))
    val xregTest = new DenseMatrix(rows = 10, cols = 2, data = Array(1.127900, 1.127100,  1.121200,  1.117100, 1.115400, 1.115400, 1.115400, 1.115400, 1.115400,
      1.127900, 39.44000,39.44000,39.44000,41.52000,41.45000,39.79000,39.46000,39.46000,39.46000,39.46000))
    val results = modelTrain.predict(tsTest, xregTest)

    results.map( value => value should be (41.0 +- 1))
  }
}