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
import org.apache.spark.mllib.linalg.DenseVector
import org.scalatest.FunSuite

class ARIMAXSuite extends FunSuite {
  // Data from http://www.robjhyndman.com/data/ - command to use this data available on website
  // robjhyndman.com/talks/RevolutionR/exercises1.pdf
  val gdp_train =  scala.io.Source.fromInputStream(getClass.getClassLoader.getResourceAsStream("data_train.csv"))
    .getLines().drop(1).map(a => a.split(",",4).map(_.trim).slice(3,4).map(va => va.toDouble)).toArray.flatten
  val sales_train =  scala.io.Source.fromInputStream(getClass.getClassLoader.getResourceAsStream("data_train.csv"))
    .getLines().drop(1).map(a => a.split(",",4).map(_.trim).slice(1,2).map(va => va.toDouble)).toArray.flatten
  val adBudget_train = scala.io.Source.fromInputStream(getClass.getClassLoader.getResourceAsStream("data_train.csv"))
    .getLines().drop(1).map(a => a.split(",",4).map(_.trim).slice(2,3).map(va => va.toDouble)).toArray.flatten

  val gdp_test =  scala.io.Source.fromInputStream(getClass.getClassLoader.getResourceAsStream("data_test.csv"))
    .getLines().drop(1).map(a => a.split(",",4).map(_.trim).slice(3,4).map(va => va.toDouble)).toArray.flatten
  val sales_test =  scala.io.Source.fromInputStream(getClass.getClassLoader.getResourceAsStream("data_test.csv"))
    .getLines().drop(1).map(a => a.split(",",4).map(_.trim).slice(1,2).map(va => va.toDouble)).toArray.flatten
  val adBudget_test = scala.io.Source.fromInputStream(getClass.getClassLoader.getResourceAsStream("data_test.csv"))
    .getLines().drop(1).map(a => a.split(",",4).map(_.trim).slice(2,3).map(va => va.toDouble)).toArray.flatten

  val tsTrain = new DenseVector(gdp_train)
  val xregTrain = new DenseMatrix(rows = sales_train.length, cols = 2, data = sales_train ++ adBudget_train)

  val tsTest = new BreezeDenseVector(gdp_test)
  val xregTest = new DenseMatrix(rows = sales_test.length, cols = 2, data = sales_test ++ adBudget_test)
  val xregTestChangedColumns = new DenseMatrix(rows = sales_test.length, cols = 2, data = adBudget_test ++ sales_test)

  val tsTrain_2 = new DenseVector(Array(93.0,82,109,110,109,84,100,91,119,78,99,92,76,99,84,103,107,106,106,89,121,103,92,94,99,94,90,99,100,125,78,95,92,84,99,88,85,121,119,94,89,121,110,110,78,88,86,77,106,127,91,98,108,110,88,118,112,104,97,100,97,96,95,111,84,102,98,110,108,92,121,104,109,105,93,74,106,118,97,109,90,91,95,95,111,112,96,122,108,96,78,124,79,89,98,127,110,92,120,109,106,124,135,110,98,108,109,103,106,92,89,82,118,94,112,86))
  val xregTrain_2 = new DenseMatrix(rows = 116, cols = 4, data = Array(416,393,444,445,426,435,471,397,454,416,424,395,401,471,400,418,476,436,442,472,492,443,418,417,423,382,433,409,436,437,372,419,423,415,432,413,361,415,437,391,395,468,415,386,410,437,401,446,492,443,438,417,384,418,403,408,380,422,432,405,437,444,485,426,411,440,400,440,432,439,431,384,404,439,401,401,427,375,411,428,376,407,403,454,478,418,428,401,467,456,446,509,406,431,458,469,450,462,538,435,485,439,451,457,495,479,418,423,430,477,423,462,481,406,450,405,
    0,0,0,0,1,1,0,0,0,0,0,1.0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,
    28,28,28,28,28,28,29,29,29,29,29,29,29,21,21,21,21,21,21,21,28,28,28,28,28,28,28,21,21,21,21,21,21,21,30,30,30,30,30,30,30,42,42,42,15,15,15,15,19,19,19,19,19,19,19,23,23,23,23,23,23,23,25,25,25,25,25,25,25,16,16,16,16,16,16,16,17,17,17,17,17,17,17,21,21,21,21,21,26,26,26,35,35,35,35,35,35,35,34,34,34,34,34,34,34,25,25,25,25,25,25,25,24,24,24,24,
    55,57,53,55,57,50,50,53,51,55,48,46,42,41,48,48,55,59,57,55,59,53,46,44,41,33,32,42,41,37,44,41,44,42,41,37,46,46,37,44,42,39,41,35,57,62,55,53,53,55,55,42,46,42,42,48,50,44,50,48,50,57,55,59,59,53,57,60,55,51,44,42,41,48,50,46,41,39,50,53,48,42,39,33,44,37,35,41,54,53,50,47,52,52,57,53,53,50,55,46,51,56,57,57,57,53,50,42,49,52,53,50,46,48,49,52))
  val tsTest_2 = new BreezeDenseVector(Array(100.0 ,98 ,102 ,98 ,112 ,99 ,99 ,87 ,103 ,115 ,101 ,125 ,117 ,109 ,111 ,105))
  val xregTest_2 = new DenseMatrix(rows = 16, cols = 4, data = Array( 465,453,472,454,432,431,475,393,437,537,462,539,471,455,466,490,
    1,1,0,0,0,0,0,1,1,0,0,0,0,0,1.0,1,
    24,24,25,25,25,25,25,25,25,23,23,23,23,23,23,23,
    51,54,49,46,42,41,45,46,48,41,42,48,43,47,48,46 ))

  def average(v: DenseVector) = {
    var sum = 0.0
    v.values.map( va => sum += va)
    println("Mean=" + sum/tsTrain.values.length)
  }
  /**
    * Moving average with xregTrain variables tests
    */
  test("1 MAX(0,0,1) 1 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 0, 1, tsTrain, xregTrain, 1, true)
    // ma1, 4xreg x2 = 9
    //    assert( model1.coefficients.length == 9)
    val results = model1.predict(tsTest, xregTest)
    val results2 = model1.predict(tsTest, xregTestChangedColumns)
    println(results)
    println(results2)
  }
  test("2 MAX(0,0,1) 1 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 0, 1, tsTrain, xregTrain, 1, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("3 MAX(0,0,1) 0 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 0, 1, tsTrain, xregTrain, 0, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("4 MAX(0,0,1) 0 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 0, 1, tsTrain, xregTrain, 0, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }

  test("5 MAX(0,0,2) 1 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 0, 2, tsTrain, xregTrain, 1, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("6 MAX(0,0,2) 1 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 0, 2, tsTrain, xregTrain, 1, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("7 MAX(0,0,2) 0 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 0, 2, tsTrain, xregTrain, 0, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("8 MAX(0,0,2) 0 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 0, 2, tsTrain, xregTrain, 0, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }

  test("9 MAX(0,0,3) 1 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 0, 3, tsTrain, xregTrain, 1, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("10 MAX(0,0,3) 1 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 0, 3, tsTrain, xregTrain, 1, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("11 MAX(0,0,3) 0 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 0, 3, tsTrain, xregTrain, 0, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("12 MAX(0,0,3) 0 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 0, 3, tsTrain, xregTrain, 0, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }

  test("13 MAX(0,1,1) 1 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 1, 1, tsTrain, xregTrain, 1, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("14 MAX(0,1,1) 1 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 1, 1, tsTrain, xregTrain, 1, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("15 MAX(0,1,1) 0 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 1, 1, tsTrain, xregTrain, 0, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("16 MAX(0,1,1) 0 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 1, 1, tsTrain, xregTrain, 0, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }

  test("17 MAX(0,1,2) 1 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 1, 2, tsTrain, xregTrain, 1, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("18 MAX(0,1,2) 1 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 1, 2, tsTrain, xregTrain, 1, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("19 MAX(0,1,2) 0 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 1, 2, tsTrain, xregTrain, 0, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("20 MAX(0,1,2) 0 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 1, 2, tsTrain, xregTrain, 0, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }

  test("21 MAX(0,1,3) 1 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 1, 3, tsTrain, xregTrain, 1, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("22 MAX(0,1,3) 1 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 1, 3, tsTrain, xregTrain, 1, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("23 MAX(0,1,3) 0 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 1, 3, tsTrain, xregTrain, 0, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("24 MAX(0,1,3) 0 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 1, 3, tsTrain, xregTrain, 0, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }

  test("25 MAX(0,2,1) 1 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 2, 1, tsTrain, xregTrain, 1, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("26 MAX(0,2,1) 1 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 2, 1, tsTrain, xregTrain, 1, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("27 MAX(0,2,1) 0 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 2, 1, tsTrain, xregTrain, 0, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("28 MAX(0,2,1) 0 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 2, 1, tsTrain, xregTrain, 0, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }

  test("29 MAX(0,2,2) 1 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 2, 2, tsTrain, xregTrain, 1, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("30 MAX(0,2,2) 1 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 2, 2, tsTrain, xregTrain, 1, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("31 MAX(0,2,2) 0 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 2, 2, tsTrain, xregTrain, 0, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("32 MAX(0,2,2) 0 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 2, 2, tsTrain, xregTrain, 0, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }

  test("33 MAX(0,2,3) 1 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 2, 3, tsTrain, xregTrain, 1, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("34 MAX(0,2,3) 1 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 2, 3, tsTrain, xregTrain, 1, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("35 MAX(0,2,3) 0 t"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 2, 3, tsTrain, xregTrain, 0, true)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
  }
  test("36 MAX(0,2,3) 0 f"){
    average(tsTrain)
    val model1 = ARIMAX.fitModel(0, 2, 3, tsTrain, xregTrain, 0, false)
    //    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest, xregTest)
    println(results)
    println("-------------END-OF-MAX-TESTS----------------")
  }

  test("1 ARIMAX(1,1,1) 0 t t"){
    // c, ar, ma, 4xreg
    val model1 = ARIMAX.fitModel(1, 1, 1, tsTrain_2, xregTrain_2, 0, true, true)
    assert( model1.coefficients.length == 7)
    val results = model1.predict(tsTest_2, xregTest_2)
  }
  test("2 ARIMAX(1,1,1) 0 t f"){
    // c, ar, ma, 4xreg
    val model2 = ARIMAX.fitModel(1, 1, 1, tsTrain_2, xregTrain_2, 0, true, false)
    assert( model2.coefficients.length == 7)
    val results = model2.predict(tsTest_2, xregTest_2)
  }
  test("3 ARIMAX(1,1,1) 0 f f"){
    // c, ar, ma
    val model3 = ARIMAX.fitModel(1, 1, 1, tsTrain_2, xregTrain_2, 0, false, false)
    assert( model3.coefficients.length == 3)
    val results = model3.predict(tsTest_2, xregTest_2)
  }
  test("4 ARIMAX(1,1,1) 0 f t"){
    // c, ar, ma
    val model4 = ARIMAX.fitModel(1, 1, 1, tsTrain_2, xregTrain_2, 0, false, true)
    assert( model4.coefficients.length == 3)
    val results = model4.predict(tsTest_2, xregTest_2)
  }
  test("5 ARIMAX(1,1,1) 1 t t"){
    // c, ar, ma, 4xreg x 2
    val model5 = ARIMAX.fitModel(1, 1, 1, tsTrain_2, xregTrain_2, 1, true, true)
    assert( model5.coefficients.length == 11)
    val results = model5.predict(tsTest_2, xregTest_2)
  }
  test("6 ARIMAX(1,1,1) 1 t f"){
    // c, ar, ma, 4xreg x 2
    val model6 = ARIMAX.fitModel(1, 1, 1, tsTrain_2, xregTrain_2, 1, true, false)
    assert( model6.coefficients.length == 11)
    val results = model6.predict(tsTest_2, xregTest_2)
  }
  test("7 ARIMAX(1,1,1) 1 f f"){
    // c, ar, ma, 4xreg
    val model7 = ARIMAX.fitModel(1, 1, 1, tsTrain_2, xregTrain_2, 1, false, false)
    assert( model7.coefficients.length == 7)
    val results = model7.predict(tsTest_2, xregTest_2)
  }
  test("8 ARIMAX(1,1,1) 1 f t"){
    // c, ar, ma, 4xreg
    val model8 = ARIMAX.fitModel(1, 1, 1, tsTrain_2, xregTrain_2, 1, false, true)
    assert( model8.coefficients.length == 7)
    val results = model8.predict(tsTest_2, xregTest_2)
  }
  val p = 2
  test(s"9 ARIMAX(2,1,1) 0 t t"){
    // c, ar, ma, 4xreg
    val model1 = ARIMAX.fitModel(p, 1, 1, tsTrain_2, xregTrain_2, 0, true, true)
    assert( model1.coefficients.length == 8)
    val results = model1.predict(tsTest_2, xregTest_2)
  }
  test(s"10 ARIMAX(2,1,1) 0 t f"){
    // c, ar, ma, 4xreg
    val model2 = ARIMAX.fitModel(p, 1, 1, tsTrain_2, xregTrain_2, 0, true, false)
    assert( model2.coefficients.length == 8)
    val results = model2.predict(tsTest_2, xregTest_2)
  }
  test(s"11 ARIMAX(2,1,1) 0 f f"){
    // c, ar, ma
    val model3 = ARIMAX.fitModel(p, 1, 1, tsTrain_2, xregTrain_2, 0, false, false)
    assert( model3.coefficients.length == 4)
    val results = model3.predict(tsTest_2, xregTest_2)
  }
  test(s"12 ARIMAX(2,1,1) 0 f t"){
    // c, ar, ma
    val model4 = ARIMAX.fitModel(p, 1, 1, tsTrain_2, xregTrain_2, 0, false, true)
    assert( model4.coefficients.length == 4)
    val results = model4.predict(tsTest_2, xregTest_2)
  }
  test(s"13 ARIMAX(2,1,1) 1 t t"){
    // c, ar, ma, 4xreg x 2
    val model5 = ARIMAX.fitModel(p, 1, 1, tsTrain_2, xregTrain_2, 1, true, true)
    assert( model5.coefficients.length == 12)
    val results = model5.predict(tsTest_2, xregTest_2)
  }
  test(s"14 ARIMAX(2,1,1) 1 t f"){
    // c, ar, ma, 4xreg x 2
    val model6 = ARIMAX.fitModel(p, 1, 1, tsTrain_2, xregTrain_2, 1, true, false)
    assert( model6.coefficients.length == 12)
    val results = model6.predict(tsTest_2, xregTest_2)
  }
  test(s"15 ARIMAX(2,1,1) 1 f f"){
    // c, ar, ma, 4xreg
    val model7 = ARIMAX.fitModel(p, 1, 1, tsTrain_2, xregTrain_2, 1, false, false)
    assert( model7.coefficients.length == 8)
    val results = model7.predict(tsTest_2, xregTest_2)
  }
  test(s"16 ARIMAX(2,1,1) 1 f t"){
    // c, ar, ma, 4xreg
    val model8 = ARIMAX.fitModel(p, 1, 1, tsTrain_2, xregTrain_2, 1, false, true)
    assert( model8.coefficients.length == 8)
    val results = model8.predict(tsTest_2, xregTest_2)
  }
  val q = 2
  test(s"17 ARIMAX(2,1,2) 0 t t"){
    // c, ar, ma, 4xreg
    val model1 = ARIMAX.fitModel(p, 1, q, tsTrain_2, xregTrain_2, 0, true, true)
    assert( model1.coefficients.length == 9)
    val results = model1.predict(tsTest_2, xregTest_2)
  }
  test(s"18 ARIMAX(2,1,2) 0 t f"){
    // c, ar, ma, 4xreg
    val model2 = ARIMAX.fitModel(p, 1, q, tsTrain_2, xregTrain_2, 0, true, false)
    assert( model2.coefficients.length == 9)
    val results = model2.predict(tsTest_2, xregTest_2)
  }
  test(s"19 ARIMAX(2,1,2) 0 f f"){
    // c, ar, ma
    val model3 = ARIMAX.fitModel(p, 1, q, tsTrain_2, xregTrain_2, 0, false, false)
    assert( model3.coefficients.length == 5)
    val results = model3.predict(tsTest_2, xregTest_2)
  }
  test(s"20 ARIMAX(2,1,2) 0 f t"){
    // c, ar, ma
    val model4 = ARIMAX.fitModel(p, 1, q, tsTrain_2, xregTrain_2, 0, false, true)
    assert( model4.coefficients.length == 5)
    val results = model4.predict(tsTest_2, xregTest_2)
  }
  test(s"21 ARIMAX(2,1,2) 1 t t"){
    // c, ar, ma, 4xreg x 2
    val model5 = ARIMAX.fitModel(p, 1, q, tsTrain_2, xregTrain_2, 1, true, true)
    assert( model5.coefficients.length == 13)
    val results = model5.predict(tsTest_2, xregTest_2)
  }
  test(s"22 ARIMAX(2,1,2) 1 t f"){
    // c, ar, ma, 4xreg x 2
    val model6 = ARIMAX.fitModel(p, 1, q, tsTrain_2, xregTrain_2, 1, true, false)
    assert( model6.coefficients.length == 13)
    val results = model6.predict(tsTest_2, xregTest_2)
  }
  test(s"23 ARIMAX(2,1,2) 1 f f"){
    // c, ar, ma, 4xreg
    val model7 = ARIMAX.fitModel(p, 1, q, tsTrain_2, xregTrain_2, 1, false, false)
    assert( model7.coefficients.length == 9)
    val results = model7.predict(tsTest_2, xregTest_2)
  }
  test(s"24 ARIMAX(2,1,2) 1 f t"){
    // c, ar, ma, 4xreg
    val model8 = ARIMAX.fitModel(p, 1, q, tsTrain_2, xregTrain_2, 1, false, true)
    assert( model8.coefficients.length == 9)
    val results = model8.predict(tsTest_2, xregTest_2)
  }
  val d = 2
  test(s"25 ARIMAX(2,2,2) 0 t t"){
    // c, ar, ma, 4xreg
    val model1 = ARIMAX.fitModel(p, d, q, tsTrain_2, xregTrain_2, 0, true, true)
    assert( model1.coefficients.length == 9)
    val results = model1.predict(tsTest_2, xregTest_2)
  }
  test(s"26 ARIMAX(2,2,2) 0 t f"){
    // c, ar, ma, 4xreg
    val model2 = ARIMAX.fitModel(p, d, q, tsTrain_2, xregTrain_2, 0, true, false)
    assert( model2.coefficients.length == 9)
    val results = model2.predict(tsTest_2, xregTest_2)
  }
  test(s"27 ARIMAX(2,2,2) 0 f f"){
    // c, ar, ma
    val model3 = ARIMAX.fitModel(p, d, q, tsTrain_2, xregTrain_2, 0, false, false)
    assert( model3.coefficients.length == 5)
    val results = model3.predict(tsTest_2, xregTest_2)
  }
  test(s"28 ARIMAX(2,2,2) 0 f t"){
    // c, ar, ma
    val model4 = ARIMAX.fitModel(p, d, q, tsTrain_2, xregTrain_2, 0, false, true)
    assert( model4.coefficients.length == 5)
    val results = model4.predict(tsTest_2, xregTest_2)
  }
  test(s"29 ARIMAX(2,2,2) 1 t t"){
    // c, ar, ma, 4xreg x 2
    val model5 = ARIMAX.fitModel(p, d, q, tsTrain_2, xregTrain_2, 1, true, true)
    assert( model5.coefficients.length == 13)
    val results = model5.predict(tsTest_2, xregTest_2)
  }
  test(s"30 ARIMAX(2,2,2) 1 t f"){
    // c, ar, ma, 4xreg x 2
    val model6 = ARIMAX.fitModel(p, d, q, tsTrain_2, xregTrain_2, 1, true, false)
    assert( model6.coefficients.length == 13)
    val results = model6.predict(tsTest_2, xregTest_2)
  }
  test(s"31 ARIMAX(2,2,2) 1 f f"){
    // c, ar, ma, 4xreg
    val model7 = ARIMAX.fitModel(p, d, q, tsTrain_2, xregTrain_2, 1, false, false)
    assert( model7.coefficients.length == 9)
    val results = model7.predict(tsTest_2, xregTest_2)
  }
  test(s"32 ARIMAX(2,2,2) 1 f t"){
    // c, ar, ma, 4xreg
    val model8 = ARIMAX.fitModel(p, d, q, tsTrain_2, xregTrain_2, 1, false, true)
    assert( model8.coefficients.length == 9)
    val results = model8.predict(tsTest_2, xregTest_2)
  }
}