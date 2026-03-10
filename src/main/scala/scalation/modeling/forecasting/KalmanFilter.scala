/*
 * KalmanFilter.scala
 * A class for implementing a Kalman Filter.
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2023 scalation.modeling.forecasting
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * @author: John Miller, Lokesh Adusumilli, Nirupom Bose Roy
 */

package scalation.modeling.forecasting

import scalation.mathstat._

/**
 * The `KalmanFilter` class provides a simple implementation of a Kalman filter.
 * It is useful for smoothing noisy data and for providing better estimates of the
 * state of a system.
 *
 * @param f The state transition matrix.
 * @param q The process noise covariance matrix.
 * @param h The measurement matrix.
 * @param r The measurement noise covariance matrix.
 * @param x The initial state vector.
 * @param p The initial covariance matrix.
 */
class KalmanFilter(
                    val f: MatrixD,
                    val q: MatrixD,
                    val h: MatrixD,
                    val r: MatrixD,
                    var x: VectorD,
                    var p: MatrixD
                  ) {
  // ... (rest of the class is unchanged) ...
  def predict(): Unit = {
    x = f * x
    p = f * p * f.transpose + q
  }

  def update(z: VectorD): Unit = {
    val y = z - h * x
    val s = h * p * h.transpose + r
    val k = p * h.transpose * s.inverse
    x = x + k * y

    val i = MatrixD.eye(p.dim, p.dim)
    val ikh = i - k * h
    p = ikh * p * ikh.transpose + k * r * k.transpose
  }

  def copyFilter(): KalmanFilter =
    new KalmanFilter(f.copy, q.copy, h.copy, r.copy, x.copy, p.copy)
}