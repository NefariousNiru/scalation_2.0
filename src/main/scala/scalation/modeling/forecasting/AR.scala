
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive (AR)
 */

package scalation
package modeling
package forecasting
import scalation.mathstat._
//import scalation.modeling.forecasting.Diagnoser
import scalation.modeling.qoF_names
//import scala.math.abs
import scalation.optimization.quasi_newton.{LBFGS_B => Optimizer}
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `AR` class provides basic time series analysis capabilities for Auto-Regressive
 *  (AR) models.  AR models are often used for forecasting.
 *  Given time series data stored in vector y, its next value y_t = combination of last p values.
 *
 *      y_t = b dot [1, y_t-1, ..., y_t-p) + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param y         the response vector (time series data) 
 *  @param hh        the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng      the time range, if relevant (time index may suffice)
 *  @param hparam    the hyper-parameters (defaults to AR.hp)
 *  @param bakcast   whether a backcasted value is prepended to the time series (defaults to false)
 *  @param adjusted  whether in `Correlogram` when calculating auto-covarainces/auto-correlations
 *                   to adjust to account for the number of elements in the sum Σ (or use dim-1)
 *                   @see `VectorD.acov`
 */
class AR (y: VectorD, hh: Int, tRng: Range = null,
          hparam: HyperParameter = AR.hp,
          bakcast: Boolean = false, adjusted: Boolean = true)
      extends Forecaster (y, hh, tRng, hparam, bakcast)
         with Correlogram (y, adjusted):

    private   val flaw = flawf ("AR")                                   // flaw function
    protected val p    = hparam("p").toInt                              // use the last p values
    protected var δ    = NO_DOUBLE                                      // drift/intercept/constant term

    modelName = s"AR($p)"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train/fit an `AR` model to the times-series data in vector y_.
     *  Estimate the coefficient vector b for a p-th order Auto-Regressive AR(p) model.
     *  Uses Durbin-Levinson Algorithm (in `Correlogram`) to determine the coefficients.
     *  The b (φ) vector is p-th row of psi matrix (ignoring the first (0th) column).
     *  @param x_null  the data/input matrix (ignored, pass null)
     *  @param y_      the training/full response vector (e.g., full y)
     */
    override def train (x_null: MatrixD, y_ : VectorD): Unit =
        makeCorrelogram (y_)                                            // correlogram computes psi matrix
        b = psiM(p)(1 until p+1).reverse                                // coefficients = p-th row, columns 1, 2, ... p
        δ = statsF.mu * (1 - b.sum)                                     // compute drift/intercept
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the parameter vector for the AR(p) model.
     */
    override def parameter: VectorD = δ +: b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *
     *      y_t = δ +  b_0 y_t-1 + b_1 y_t-2 + ... + b_p-1 y_t-p
     *
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions
     */
    override def predict (t: Int, y_ : VectorD): Double =
//      δ + rdot (b, y_, t-1)                                          // old way
        val x = y_.prior (p, t)                                        // x = [ y_t-p, ... y_t-1 ]
        δ + (b dot x)
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forge a vector from va;ues in the FORECAST MATRIX yf. 
     *  @patam yf  the forecast matrix
     *  @param t   the time point from which to make forecasts
     *  @param h   the forecasting horizon
     */
    def forge (yf: MatrixD, t: Int, h: Int): VectorD =
        val x_act   = yf(?, 0).prior (max0 (p-h+1), t)                 // get actual lagged y-values (endogenous)
        val x_fcast = yf(t)(1 until h)                                 // get forecasted y-values
        x_act ++ x_fcast
    end forge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a vector of size hh, h = 1 to hh-steps ahead forecasts for the model,
     *  i.e., forecast the following time points:  t+1, ..., t+h.
     *  Intended to work with rolling validation (analog of predict method).
     *  @param t   the time point from which to make forecasts
     *  @param y_  the actual values to use in making predictions
     */
    override def forecast (t: Int, y_ : VectorD = yb): VectorD =
        val yh = new VectorD (hh)                                       // hold forecasts for each horizon
        for h <- 1 to hh do
//          val pred = δ + rdot (b, yf, t, h-1)                         // slide in prior forecasted values
            val x = forge (yf, t, h)
            val pred = δ + (b dot x)
            yf(t, h) = pred                                             // record in forecast matrix
            yh(h-1)  = pred                                             // record forecasts for each horizon
        yh                                                              // return forecasts for all horizons
    end forecast

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points at horizon h (h-steps ahead).
     *  Assign into FORECAST MATRIX and return the h-steps ahead forecast.
     *  Note, `predictAll` provides predictions for h = 1.
     *  @see
     *
     *  `forecastAll` method in `Forecaster` trait.
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     *  @param y_  the actual values to use in making forecasts
     */
    override def forecastAt (h: Int, y_ : VectorD = yb): VectorD =
        if h < 2 then flaw ("forecastAt", s"horizon h = $h must be at least 2")

        for t <- y_.indices do                                          // make forecasts over all time points for horizon h
//          yf(t, h) = δ + rdot (b, yf, t, h-1)                         // record in forecast matrix (old way)
            val x = forge (yf, t, h)
            yf(t, h) = δ + (b dot x)                                    // record in forecast matrix
        yf(?, h)                                                        // return the h-step ahead forecast vector
    end forecastAt

end AR


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `AR` companion object provides factory methods for the
 *  `AR` class.
 */
object AR:

    /** Base hyper-parameter specification for the `AR`, ARMA, ARIMA, `SARIMA` and  `SARIMAX` classes
     */
    val hp = new HyperParameter
    hp += ("p", 1, 1)                               // number of Auto-Regressive (AR) parameters
    hp += ("d", 1, 1)                               // number of Differences to take
    hp += ("q", 1, 1)                               // number of Moving-Average (MA) parameters
    hp += ("P", 1, 1)                               // number of Seasonal Auto-Regressive (AR) parameters
    hp += ("D", 1, 1)                               // number of Seasonal Differences to take
    hp += ("Q", 1, 1)                               // number of Seasonal Moving-Average (MA) parameters
    hp += ("s", 7, 7)                               // length of the Seasonal Period
    hp += ("a", 1, 1)                               // the first lag for the eXogenous variables
    hp += ("b", 2, 2)                               // the last lag for the eXogenous variables

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `AR` object.
     *  @param y       the response vector (time series data)
     *  @param hh      the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng    the time range, if relevant (time index may suffice)
     *  @param hparam  the hyper-parameters
     */
    def apply (y: VectorD, hh: Int, tRng: Range = null, hparam: HyperParameter = hp): AR =
        new AR (y, hh, tRng, hparam)
    end apply

end AR

import Example_Covid.loadData_y
import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRTest` main function tests the `AR` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRTest
 */
@main def aRTest (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new AR (y, hh)                                              // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
//  Forecaster.evalForecasts (mod, mod.getYb, hh)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end aRTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRTest2` main function tests the `AR` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRTest2
 */
@main def aRTest2 (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new AR (y, hh)                                              // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.rollValidate ()                                                 // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end aRTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRTest3` main function tests the `AR` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRTest3
 */
@main def aRTest3 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    for p <- 1 to 6 do
        AR.hp("p") = p                                                    // number of AR terms
        val mod = new AR (y, hh)                                          // create model for time series data
//      val mod = new AR (y, hh, adjusted = false)                        // use conventional rho estimation
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest ()()                                               // train and test on full dataset

//      mod.setSkip (p)                                                   // full AR-formula available when t >= p
        mod.forecastAll ()                                                // forecast h-steps ahead (h = 1 to hh) for all y
        mod.diagnoseAll (y, mod.getYf)
//      Forecaster.evalForecasts (mod, mod.getYb, hh)
//      println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
//      println (s"Final In-ST Forecast Matrix yf = ${mod.getYf.shiftDiag}")
    end for

end aRTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRTest4` main function tests the `AR` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRTest4
 */
@main def aRTest4 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    for p <- 5 to 5 do
        AR.hp("p") = p                                                    // number of AR terms
        val mod = new AR (y, hh)                                          // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
//      mod.setSkip (p)                                                   // may wish to skip until all p past values are available
        mod.trainNtest ()()

        mod.setSkip (0)                                                   // can use values from training set to not skip any in test
        mod.rollValidate ()                                               // TnT with Rolling Validation
        println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")
    end for

end aRTest4


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRTest5` main function tests the `AR` class on small dataset.
 *  Test forecasts (h = 1 step ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRTest5
 */
@main def aRTest5 (): Unit =

    val y  = VectorD (1, 3, 4, 2, 5, 7, 9, 8, 6, 3)

    var mod = new AR (y, 1)                                               // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on a Small Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
    new Baseline (y, "AR1")

    AR.hp ("p") = 2
    mod = new AR (y, 1)                                                   // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on a Small Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
    new Baseline (y, "AR2")

end aRTest5

/**
 * The `aRTest_Kalman` object is a new test case for the `AR` class.
 * It demonstrates how to use a Kalman Filter to implement an AR(1) model for forecasting.
 * The state of the Kalman filter represents the value of the time series.
 * The state transition model is the AR(1) equation itself.
 *
 * Steps:
 * 1. An AR(1) model is first trained on the data to find the autoregressive
 * coefficient (phi) and the noise variance (sigma).
 * 2. These parameters are then used to set up the KalmanFilter's matrices.
 * 3. The filter processes the data sequentially.
 * 4. A one-step-ahead forecast is made using the filter's predict method.
 */
object aRTest_Kalman extends App
{
  import scalation.mathstat.VectorD

  // A simple time series dataset
  val y = VectorD (26.6, 27.1, 27.5, 26.9, 27.2, 27.6, 27.9, 28.3, 27.8, 28.1,
    28.6, 28.8, 29.1, 28.9, 29.2, 29.3, 29.5, 29.9, 30.1, 30.4,
    30.6, 31.0, 31.3, 31.7, 31.5, 31.9, 32.2, 32.5, 32.9, 32.6)

  val h = 1                                               // forecast horizon

  println (s"Test AR with Kalman Filter on y with horizon h = $h")

  // 1. Fit a standard AR(1) model to get parameters
  val mod = AR (y, 1)
  mod.train (null, y)
  val phi = mod.parameter(1)                              // AR(1) coefficient
  val sigma2 = mod.test(null, y)._2.variance              // Variance of residuals (process noise)

  println (s"AR(1) parameters: phi = $phi, sigma^2 = $sigma2")

  // 2. Setup the Kalman Filter
  // F: State transition matrix, based on the AR(1) coefficient phi
  // Q: Process noise covariance, from the variance of the AR model's residuals
  // H: Measurement matrix, a simple 1-to-1 mapping
  // R: Measurement noise covariance, a tuning parameter (start with a small value)
  // x: Initial state, set to the first data point
  // P: Initial state covariance, start with identity
  val kf = new KalmanFilter (
    f = MatrixD ((1, 1), phi),
    q = MatrixD ((1, 1), sigma2),
    h = MatrixD ((1, 1), 1.0),
    r = MatrixD ((1,1), 0.1),
    x = VectorD (y(0)),
    p = MatrixD ((1, 1), 1.0)
  )

  // 3. Process the time series data with the filter
  for t <- 1 until y.dim do
    kf.predict ()
    kf.update (VectorD (y(t)))
  end for

  // 4. Make a one-step-ahead forecast
  kf.predict ()
  val forecast = kf.x(0)

  println (s"Forecast for y(t+1) using Kalman Filter: $forecast")
  println (s"Actual value of y(t): ${y.last}")

} // aRTest_Kalman

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRTest4_Kalman` main function tests the `AR` class on real data:
 * Forecasting COVID-19 using a Kalman Filter with Train-n-Test Split (TnT)
 * and Rolling Validation. This serves as a comparison to the standard `aRTest4`.
 * > runMain scalation.modeling.forecasting.aRTest4_Kalman
 */
/*
@main def aRTest4_Kalman (): Unit =

  val yy = loadData_y ()
  val y  = yy(0 until 116)                                              // clip the flat end
  val hh = 6                                                            // maximum forecasting horizon
  val t_s = 80                                                          // training/test split time

  // We focus on p=1, as higher-order models are prone to overfitting on this dataset,
  // leading to poor forecasts and negative F-statistics.
  val p = 1
  AR.hp("p") = p
  val mod = new AR (y, hh)
  banner (s"TnT Kalman Forecasts: ${mod.modelName} on COVID-19 Dataset")

  // === Train the base AR model on the training set ===
  val y_train = y(0 until t_s)
  mod.train(null, y_train)
  val phi = mod.parameter(1)

  val mod_test = new AR(y_train, hh)
  mod_test.train(null, y_train)
  val sigma2 = mod_test.test(null, y_train)._2.variance
  println(s"AR($p) parameters: phi = $phi, sigma^2 = $sigma2")

  // === Initialize the Kalman Filter ===
  val kf = new KalmanFilter(
    f = MatrixD((1, 1), phi),
    q = MatrixD((1, 1), sigma2),
    h = MatrixD((1, 1), 1.0),
    r = MatrixD((1, 1), 0.1),       // Measurement noise is a tuning parameter
    x = VectorD(y_train(0)),
    p = MatrixD((1, 1), 1.0)
  )

  // "Burn-in" the filter with the training data
  for t <- 1 until y_train.dim do
    kf.predict()
    kf.update(VectorD(y_train(t)))
  end for

  // === Rolling Validation using the Kalman Filter ===
  println ("Rolling Validation ...")
  val yf = new MatrixD (y.dim, hh + 1)
  yf(?, 0) = y                                // column 0 is the actual y-values

  for t <- t_s until y.dim do                 // iterate through the test set
    val y_test_actual = y(t)

    // Make a 1-step ahead forecast
    kf.predict()
    val forecast1 = kf.x(0)
    yf(t, 1) = forecast1

    // Update the filter with the actual observed value
    kf.update(VectorD(y_test_actual))
  end for

  println (s"Final TnT Kalman Forecast Matrix yf (h=1) = ${yf(?, 1)}")

  // The diagnose method may produce an error if the model's forecasts are very poor,
  // resulting in a negative F-statistic. We wrap it in a try-catch block to handle this.
  try
    mod.diagnose(y(t_s until y.dim), yf(t_s until y.dim, 1))
  catch
    case ex: Throwable => println(s"Could not generate diagnostics: ${ex.getMessage}")
  end try

end aRTest4_Kalman */

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRTest4_Kalman` main function tests the `AR` class on real data:
 * Forecasting COVID-19 using a Kalman Filter with Train-n-Test Split (TnT)
 * and Rolling Validation. This serves as a comparison to the standard `aRTest4`.
 * > runMain scalation.modeling.forecasting.aRTest4_Kalman
 */
@main def aRTest4_Kalman (): Unit =

  val yy = loadData_y ()
  val y  = yy(0 until 116)                                              // clip the flat end
  val hh = 6                                                            // maximum forecasting horizon
  val t_s = 92                                                         // training/test split time

  val p = 1
  AR.hp("p") = p
  val mod = new AR (y, hh)
  banner (s"TnT Kalman Forecasts: ${mod.modelName} on COVID-19 Dataset")

  // === Train the base AR model on the training set ===
  val y_train = y(0 until t_s)
  mod.train(null, y_train)
  val phi = mod.parameter(1)

  val mod_test = new AR(y_train, hh)
  mod_test.train(null, y_train)
  val sigma2 = mod_test.test(null, y_train)._2.variance
  println(s"AR($p) parameters: phi = $phi, sigma^2 = $sigma2")

  // === Initialize the Kalman Filter ===
  val kf = new KalmanFilter(
    f = MatrixD((1, 1), phi),
    q = MatrixD((1, 1), sigma2),
    h = MatrixD((1, 1), 1.0),
    r = MatrixD((1, 1), 0.1),       // Measurement noise is a tuning parameter
    x = VectorD(y_train(0)),
    p = MatrixD((1, 1), 1.0)
  )

  // "Burn-in" the filter with the training data
  for t <- 1 until y_train.dim do
    kf.predict()
    kf.update(VectorD(y_train(t)))
  end for

  // === Rolling Validation using the Kalman Filter ===
  println ("Rolling Validation ...")
  val yf = new MatrixD (y.dim, hh + 1)
  yf(?, 0) = y                                // column 0 is the actual y-values

  for t <- t_s until y.dim do                 // iterate through the test set
    val y_test_actual = y(t)

    // Make a 1-step ahead forecast
    kf.predict()
    val forecast1 = kf.x(0)
    yf(t, 1) = forecast1

    // Update the filter with the actual observed value from the test set
    kf.update(VectorD(y_test_actual))
  end for

  println (s"Final TnT Kalman Forecast Matrix yf (h=1) = ${yf(?, 1)}")

  // --- Print the Model Summary ---
  println("\n--- Model Summary for Kalman Filter Forecasts ---")
  val y_test = y(t_s until y.dim)
  val y_pred = yf(t_s until y.dim, 1)
  mod.diagnose(y_test, y_pred)

end aRTest4_Kalman

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aR_KalmanEstimationTest` object demonstrates how to estimate the parameters
 * of an AR(2) model using the Kalman Filter for Maximum Likelihood Estimation (MLE).
 * This approach does NOT use the conventional AR training methods.
 *
 * The process works as follows:
 * 1. Define an objective function (e.g., `negativeLogLikelihood`) that takes a
 * vector of model parameters (phi_1, phi_2, sigma2) as input.
 * 2. Inside this function, configure a Kalman Filter in its state-space representation
 * for an AR(2) process.
 * 3. Iterate through the data using the filter's predict-update cycle. At each step,
 * use the filter's prediction error and prediction error covariance to compute
 * the log-likelihood for that data point.
 * 4. Sum the log-likelihoods over all data points. Return the negative of this sum
 * (since optimizers typically minimize).
 * 5. Use a numerical optimizer (like L-BFGS-B) to find the parameter vector that
 * minimizes the negative log-likelihood, thus giving us our MLE estimates.
 *
 * > runMain scalation.modeling.forecasting.aR_KalmanEstimationTest
 */
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 * @version 2.0
 * @date    Sun Jun 30 13:27:00 EDT 2024
 * @see     LICENSE (MIT style license file).
 *
 * @note    Model: Auto-Regressive (AR)
 */
/*

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aR_KalmanEstimationTest` object demonstrates how to estimate the parameters
 * of a general AR(p) model using the Kalman Filter for Maximum Likelihood Estimation (MLE).
 * This version is dynamic based on the number of lags 'p'.
 * > runMain scalation.modeling.forecasting.aR_KalmanEstimationTest
 */
@main def aR_KalmanEstimationTest (): Unit =

  // --- DYNAMIC PARAMETER: Set the desired number of AR lags here ---
  val p = 5 // Example: Set lags to 5

  // Load and prepare the COVID-19 time series data.
  val yy = loadData_y ()
  val y  = yy(0 until 92)
  banner(s"Estimating AR($p) parameters for COVID-19 using pure Kalman Filter MLE")

  // A simple helper class to calculate and report quality of fit metrics.
  class Fit(y: VectorD, yp: VectorD):
    private val e = y - yp
    def mae: Double = e.abs.mean
    def rmse: Double = math.sqrt(e.normSq / e.dim)
    def smape: Double = (e.abs / (y.abs + yp.abs)).sum * 100.0 / y.dim
    def report(): Unit =
      println(s"sMAPE = $smape")
      println(s"MAE   = $mae")
      println(s"RMSE  = $rmse")
    end report
  end Fit

  /**
   * The objective function for the optimizer.
   * @param b The vector of parameters: [phi_1, ..., phi_p, sigma2(Q), r_noise(R)]
   */
  def negativeLogLikelihood (b: VectorD): Double =
    val phis      = b(0 until p)
    val sig2_proc = b(p)
    val sig2_obs  = b(p + 1)

    if sig2_proc <= 0 || sig2_obs <= 0 then return Double.PositiveInfinity

    val f = new MatrixD(p, p)
    f(0) = phis
    for i <- 1 until p do f(i, i - 1) = 1.0

    val q = new MatrixD(p, p); q(0, 0) = sig2_proc
    val h = new MatrixD(1, p); h(0, 0) = 1.0
    val r = MatrixD ((1, 1), sig2_obs)

    val x0 = y(0 until p).reverse
    val p0 = MatrixD.eye(p, p) * 1.0
    val kf = new KalmanFilter(f, q, h, r, x0, p0)
    var logLik = 0.0

    for t <- p until y.dim do
      kf.predict()
      val z = VectorD(y(t))
      val y_err = z - kf.h * kf.x
      val s = kf.h * kf.p * kf.h.transpose + kf.r
      val err_sq_term = y_err(0) * (1.0 / s(0, 0)) * y_err(0)
      logLik += -0.5 * (math.log(2 * math.Pi) + math.log(s(0, 0)) + err_sq_term)
      kf.update(z)
    end for
    -logLik
  end negativeLogLikelihood

  // --- DYNAMIC OPTIMIZER SETUP ---
  val num_params = p + 2
  val b0 = new VectorD(num_params)
  b0(p) = y.variance
  b0(p + 1) = y.variance * 0.1

  val lowerBounds = VectorD.fill(p)(-2.0) ++ VectorD(1e-6, 1e-6)
  val upperBounds = VectorD.fill(p)(2.0)  ++ VectorD(Double.PositiveInfinity, Double.PositiveInfinity)
  val optimizer = new Optimizer(f = negativeLogLikelihood, l_u = (lowerBounds, upperBounds))

  // --- Solve and Report ---
  val (est_loss, est_params) = optimizer.solve(b0)

  println("\nEstimation Complete:")
  println(s"Final Negative Log-Likelihood: $est_loss")
  for i <- 0 until p do println(s"Estimated phi_${i+1}:   ${est_params(i)}")
  println(s"Estimated sigma^2 (Q): ${est_params(p)}")
  println(s"Estimated R:         ${est_params(p + 1)}")

  // --- Final Model Evaluation ---
  println(s"\n--- Quality of Fit for Kalman-Estimated AR($p) Model (skipping first $p values) ---")

  val b_kalman = est_params(0 until p)
  val delta_kalman = y.mean * (1 - b_kalman.sum)

  def predict_kalman(t: Int, y_hist: VectorD): Double =
    // For the first p values, predictions are not well-defined, so we use the actual values.
    if t < p then y_hist(t) else delta_kalman + (b_kalman dot y_hist.prior(p, t))
  end predict_kalman

  val yp_kalman = new VectorD(y.dim)
  for t <- y.indices do yp_kalman(t) = predict_kalman(t, y)

  // FIX: Create slices of the vectors to exclude the first p initial values from the fit calculation.
  val y_eval  = y(p until y.dim)
  val yp_eval = yp_kalman(p until y.dim)

  // Calculate and print the final performance metrics on the evaluation slice.
  val kalman_fit = new Fit(y_eval, yp_eval)
  kalman_fit.report()

  // --- Plot the results ---
  // The plot will still show all points for context, but the metrics are calculated on the sliced data.
  val t = VectorD.range(0, y.dim)
  new Plot(t, y, yp_kalman, s"Kalman-Estimated AR($p) vs. Actual", lines = true)

end aR_KalmanEstimationTest


@main def aR_KalmanEstimationTest (): Unit =

  // Loop through lags 1 to 5 to benchmark
  for p <- 6 to 6 do

    val yy = loadData_y ()
    val y  = yy(0 until 116)
    banner(s"Estimating AR($p) parameters for COVID-19 using pure Kalman Filter MLE")

    class Fit(y: VectorD, yp: VectorD):
      private val e = y - yp                                       // Calculate the error/residuals.
      private val sse = e.normSq                                   // Sum of Squared Errors.
      private val sst = (y - y.mean).normSq                        // Total Sum of Squares.

      def mae: Double = e.abs.mean                                 // Mean Absolute Error.
      def rmse: Double = math.sqrt(sse / e.dim)                    // Root Mean Squared Error.
      def smape: Double = (e.abs / (y.abs + yp.abs)).sum * 100.0 / y.dim // Symmetric Mean Absolute Percentage Error.
      def r_squared: Double = 1.0 - (sse / sst)                    // R-squared (Coefficient of Determination).

      // Print all the calculated metrics.
      def report(): Unit =
        println(s"sMAPE = $smape")
        println(s"MAE   = $mae")
        println(s"RMSE  = $rmse")
        println(s"R^2   = $r_squared")
      end report
    end Fit

    def negativeLogLikelihood (b: VectorD): Double =
      val phis      = b(0 until p)
      val sig2_proc = b(p)
      val sig2_obs  = b(p + 1)

      if sig2_proc <= 0 || sig2_obs <= 0 then return Double.PositiveInfinity

      val f = new MatrixD(p, p); f(0) = phis; for i <- 1 until p do f(i, i - 1) = 1.0
      val q = new MatrixD(p, p); q(0, 0) = sig2_proc
      val h = new MatrixD(1, p); h(0, 0) = 1.0
      val r = MatrixD ((1, 1), sig2_obs)

      val x0 = y(0 until p).reverse
      val p0 = MatrixD.eye(p, p) * 1.0
      val kf = new KalmanFilter(f, q, h, r, x0, p0)
      var logLik = 0.0

      for t <- p until y.dim do
        kf.predict()
        val z = VectorD(y(t))
        val y_err = z - kf.h * kf.x
        val s = kf.h * kf.p * kf.h.transpose + kf.r
        val err_sq_term = y_err(0) * (1.0 / s(0, 0)) * y_err(0)
        logLik += -0.5 * (math.log(2 * math.Pi) + math.log(s(0, 0)) + err_sq_term)
        kf.update(z)
      end for
      -logLik
    end negativeLogLikelihood

    // --- DYNAMIC OPTIMIZER SETUP ---
    // Reverted to a naive initial guess, without the two-stage fitting.
    val num_params = p + 2
    val b0 = new VectorD(num_params)
    b0(p) = y.variance
    b0(p + 1) = y.variance * 0.1

    val lowerBounds = VectorD.fill(p)(-2.0) ++ VectorD(1e-6, 1e-6)
    val upperBounds = VectorD.fill(p)(2.0)  ++ VectorD(Double.PositiveInfinity, Double.PositiveInfinity)
    val optimizer = new Optimizer(f = negativeLogLikelihood, l_u = (lowerBounds, upperBounds))

    // --- Solve and Report ---
    val (est_loss, est_params) = optimizer.solve(b0)

    println("\nEstimation Complete:")
    for i <- 0 until p do println(s"Estimated phi_${i+1}:   ${est_params(i)}")
    println(s"Estimated sigma^2 (Q): ${est_params(p)}")
    println(s"Estimated R:         ${est_params(p + 1)}")

    // --- Final Model Evaluation ---
    println(s"\n--- Quality of Fit for Kalman-Estimated AR($p) Model (skipping first $p values) ---")

    val b_kalman = est_params(0 until p)
    val delta_kalman = y.mean * (1 - b_kalman.sum)

    def predict_kalman(t: Int, y_hist: VectorD): Double =
      if t < p then y_hist(t)
      else delta_kalman + (b_kalman dot y_hist.prior(p, t).reverse)
    end predict_kalman

    val yp_kalman = new VectorD(y.dim)
    for t <- y.indices do yp_kalman(t) = predict_kalman(t, y)

    val y_eval  = y(p until y.dim)
    val yp_eval = yp_kalman(p until y.dim)

    val kalman_fit = new Fit(y_eval, yp_eval)
    kalman_fit.report()

  end for
end aR_KalmanEstimationTest


@main def aR_KalmanRoll (): Unit =

  // --- SETUP: Define model order, data, and train/test split ---
  val p = 3                                               // Set the desired number of AR lags
  val yy = loadData_y ()
  val y_full = yy(0 until 116)
  val t_s = 92                                            // Training/test split time
  val y_train = y_full(0 until t_s)
  val y_test = y_full(t_s until y_full.dim)

  banner(s"Rolling Forecast for AR($p) on COVID-19 Data")
  println(s"Training on ${y_train.dim} points, Testing on ${y_test.dim} points.")

  // --- Helper class for evaluation metrics ---
  class Fit(y: VectorD, yp: VectorD):
    private val e = y - yp
    def mae: Double = e.abs.mean
    def rmse: Double = math.sqrt(e.normSq / e.dim)
    def smape: Double = (e.abs / (y.abs + yp.abs)).sum * 100.0 / y.dim
    def report(): Unit =
      println(s"sMAPE = $smape")
      println(s"y.dim   = ${y.dim} ")
      println(s"MAE   = $mae")
      println(s"RMSE  = $rmse")
    end report
  end Fit

  // --- TRAINING PHASE ---

  // The objective function for the optimizer (operates only on training data).
  def negativeLogLikelihood (b: VectorD): Double =
    val phis      = b(0 until p)
    val sig2_proc = b(p)
    val sig2_obs  = b(p + 1)

    if sig2_proc <= 0 || sig2_obs <= 0 then return Double.PositiveInfinity

    val f = new MatrixD(p, p); f(0) = phis; for i <- 1 until p do f(i, i - 1) = 1.0
    val q = new MatrixD(p, p); q(0, 0) = sig2_proc
    val h = new MatrixD(1, p); h(0, 0) = 1.0
    val r = MatrixD ((1, 1), sig2_obs)

    val x0 = y_train(0 until p).reverse
    val p0 = MatrixD.eye(p, p) * 1.0
    val kf = new KalmanFilter(f, q, h, r, x0, p0)
    var logLik = 0.0

    for t <- p until y_train.dim do
      kf.predict()
      val z = VectorD(y_train(t))
      val y_err = z - kf.h * kf.x
      val s = kf.h * kf.p * kf.h.transpose + kf.r
      val err_sq_term = y_err(0) * (1.0 / s(0, 0)) * y_err(0)
      logLik += -0.5 * (math.log(2 * math.Pi) + math.log(s(0, 0)) + err_sq_term)
      kf.update(z)
    end for
    -logLik
  end negativeLogLikelihood

  // Optimize to find the best parameters on the training set.
  val num_params = p + 2
  val b0 = new VectorD(num_params); b0(p) = y_train.variance; b0(p + 1) = y_train.variance * 0.1
  val lowerBounds = VectorD.fill(p)(-2.0) ++ VectorD(1e-6, 1e-6)
  val upperBounds = VectorD.fill(p)(2.0)  ++ VectorD(Double.PositiveInfinity, Double.PositiveInfinity)
  val optimizer = new Optimizer(f = negativeLogLikelihood, l_u = (lowerBounds, upperBounds))
  val (est_loss, est_params) = optimizer.solve(b0)

  println("\n--- Training Complete ---")
  for i <- 0 until p do println(s"Estimated phi_${i+1}:   ${est_params(i)}")
  println(s"Estimated sigma^2 (Q): ${est_params(p)}")
  println(s"Estimated R:         ${est_params(p + 1)}")

  // --- FORECASTING PHASE (ROLLING VALIDATION) ---

  println("\n--- Performing Rolling Forecast on Test Set ---")

  // Initialize a new Kalman filter with the parameters learned from the training data.
  val phis_final = est_params(0 until p)
  val q_final = new MatrixD(p, p); q_final(0, 0) = est_params(p)
  val r_final = MatrixD((1, 1), est_params(p + 1))
  val f_final = new MatrixD(p, p); f_final(0) = phis_final; for i <- 1 until p do f_final(i, i - 1) = 1.0
  val h_final = new MatrixD(1, p); h_final(0, 0) = 1.0

  // "Burn-in" the filter: run it over the entire training set to get the state right before the test set.
  val x_start_test = y_train(y_train.dim - p until y_train.dim).reverse
  val p_start_test = MatrixD.eye(p, p) * 1.0
  val kf_forecast = new KalmanFilter(f_final, q_final, h_final, r_final, x_start_test, p_start_test)

  // Perform the rolling forecast.
  val yp_test = new VectorD(y_test.dim)
  for t <- 0 until y_test.dim do
    kf_forecast.predict()                     // Predict the value at the next time step.
    yp_test(t) = kf_forecast.x(0)             // Store the one-step-ahead forecast.
    kf_forecast.update(VectorD(y_test(t)))    // Update the filter with the actual observed value.
  end for

  // --- EVALUATION PHASE ---

  println("\n--- Out-of-Sample Forecast Evaluation ---")
  val forecast_fit = new Fit(y_test, yp_test)
  forecast_fit.report()

  // Plot the test set results.
  val t_test = VectorD.range(t_s, y_full.dim)
  new Plot(t_test, y_test, yp_test, s"Kalman AR($p) Rolling Forecast vs. Actual", lines = true)

end aR_KalmanRoll
*/
@main def aR_KalmanEstimationTest (): Unit =

  // Loop through lags 1 to 6 to benchmark
  for p <- 1 to 6 do

    val yy = loadData_y ()
    val y  = yy(0 until 116)
    banner(s"Estimating AR($p) parameters for COVID-19 using pure Kalman Filter MLE")

    def negativeLogLikelihood (b: VectorD): Double =
      val phis      = b(0 until p)
      val sig2_proc = b(p)
      val sig2_obs  = b(p + 1)

      if sig2_proc <= 0 || sig2_obs <= 0 then return Double.PositiveInfinity

      val f = new MatrixD(p, p); f(0) = phis; for i <- 1 until p do f(i, i - 1) = 1.0
      val q = new MatrixD(p, p); q(0, 0) = sig2_proc
      val h = new MatrixD(1, p); h(0, 0) = 1.0
      val r = MatrixD ((1, 1), sig2_obs)

      val x0 = y(0 until p).reverse
      val p0 = MatrixD.eye(p, p) * 1.0
      val kf = new KalmanFilter(f, q, h, r, x0, p0)
      var logLik = 0.0

      for t <- p until y.dim do
        kf.predict()
        val z = VectorD(y(t))
        val y_err = z - kf.h * kf.x
        val s = kf.h * kf.p * kf.h.transpose + kf.r
        val err_sq_term = y_err(0) * (1.0 / s(0, 0)) * y_err(0)
        logLik += -0.5 * (math.log(2 * math.Pi) + math.log(s(0, 0)) + err_sq_term)
        kf.update(z)
      end for
      -logLik
    end negativeLogLikelihood

    // --- DYNAMIC OPTIMIZER SETUP ---
    val num_params = p + 2
    val b0 = new VectorD(num_params)
    b0(p) = y.variance
    b0(p + 1) = y.variance * 0.1

    val lowerBounds = VectorD.fill(p)(-2.0) ++ VectorD(1e-6, 1e-6)
    val upperBounds = VectorD.fill(p)(2.0)  ++ VectorD(Double.PositiveInfinity, Double.PositiveInfinity)
    val optimizer = new Optimizer(f = negativeLogLikelihood, l_u = (lowerBounds, upperBounds))

    // --- Solve and Report ---
    val (est_loss, est_params) = optimizer.solve(b0)

    println("\nEstimation Complete:")
    for i <- 0 until p do println(s"Estimated phi_${i+1}:   ${est_params(i)}")
    println(s"Estimated sigma^2 (Q): ${est_params(p)}")
    println(s"Estimated R:         ${est_params(p + 1)}")

    // --- Final Model Evaluation ---
    println(s"\n--- Quality of Fit for Kalman-Estimated AR($p) Model (skipping first $p values) ---")

    val b_kalman = est_params(0 until p)
    val delta_kalman = y.mean * (1 - b_kalman.sum)

    def predict_kalman(t: Int, y_hist: VectorD): Double =
      if t < p then y_hist(t)
      else delta_kalman + (b_kalman dot y_hist.prior(p, t).reverse)
    end predict_kalman

    val yp_kalman = new VectorD(y.dim)
    for t <- y.indices do yp_kalman(t) = predict_kalman(t, y)

    val y_eval  = y(p until y.dim)
    val yp_eval = yp_kalman(p until y.dim)

    // FIX 1: The Diagnoser class now correctly passes parameters to the parent trait constructor using 'df'.
    class KalmanDiagnoser(n_params: Int, n_obs: Int)
      extends Diagnoser(dfm = n_params.toDouble, df = (n_obs - n_params).toDouble):
      val modName = s"Kalman-AR($n_params)"
    end KalmanDiagnoser

    val reporter = new KalmanDiagnoser(p, y_eval.dim)
    val stats = reporter.diagnose(y_eval, yp_eval)

    // using the 'qoF_names' from the modeling package.
    println(s"Validation metrics for ${reporter.modName}:")
    for i <- stats.indices do
      if qoF_names(i) != "NA" then println(f"${qoF_names(i)}%10s = ${stats(i)}%12.6f")
    end for

  end for
end aR_KalmanEstimationTest

// ILI code

import scala.io.Source

@main def aR_KalmanEstimationTest_ILI (): Unit =
  //println(s"Current Working Directory: ${System.getProperty("user.dir")}")
  // 1. LOAD DATA (ILITOTAL from Column 7)
  val source = Source.fromFile("data/ILINetNov25.csv")
  val lines = source.getLines().drop(1).toVector // Skip header
  source.close()
  // Column 7 is ILITOTAL based on CSV structure
  val y = VectorD(lines.map(_.split(",")(7).toDouble))

  // 2. TRAIN / TEST SPLIT (80/20)
  val train_size = (y.dim * 0.80).toInt
  val y_train    = y(0 until train_size)
  val y_test     = y(train_size until y.dim)

  println(s"Data Loaded. Total: ${y.dim}, Train: ${y_train.dim}, Test: ${y_test.dim}")

  // Loop through lags 1 to 6 to benchmark
  for p <- 1 to 6 do

    banner(s"Estimating AR($p) parameters for ILITOTAL (Train 80%) using pure Kalman Filter MLE")

    // --- OPTIMIZATION (Strictly on Training Data) ---
    def negativeLogLikelihood (b: VectorD): Double =
      val phis      = b(0 until p)
      val sig2_proc = b(p)
      val sig2_obs  = b(p + 1)

      if sig2_proc <= 0 || sig2_obs <= 0 then return Double.PositiveInfinity

      val f = new MatrixD(p, p); f(0) = phis; for i <- 1 until p do f(i, i - 1) = 1.0
      val q = new MatrixD(p, p); q(0, 0) = sig2_proc
      val h = new MatrixD(1, p); h(0, 0) = 1.0
      val r = MatrixD ((1, 1), sig2_obs)

      // Initialize state using the beginning of Training Data
      val x0 = y_train(0 until p).reverse
      val p0 = MatrixD.eye(p, p) * 1.0
      val kf = new KalmanFilter(f, q, h, r, x0, p0)
      var logLik = 0.0

      // Iterate through TRAINING set
      for t <- p until y_train.dim do
        kf.predict()
        val z = VectorD(y_train(t))
        val y_err = z - kf.h * kf.x
        val s = kf.h * kf.p * kf.h.transpose + kf.r
        val err_sq_term = y_err(0) * (1.0 / s(0, 0)) * y_err(0)
        logLik += -0.5 * (math.log(2 * math.Pi) + math.log(s(0, 0)) + err_sq_term)
        kf.update(z)
      end for
      -logLik
    end negativeLogLikelihood

    // --- DYNAMIC OPTIMIZER SETUP ---
    val num_params = p + 2
    val b0 = new VectorD(num_params)
    // Initialize using Training Statistics to avoid leakage
    b0(p) = y_train.variance
    b0(p + 1) = y_train.variance * 0.1

    val lowerBounds = VectorD.fill(p)(-2.0) ++ VectorD(1e-6, 1e-6)
    val upperBounds = VectorD.fill(p)(2.0)  ++ VectorD(Double.PositiveInfinity, Double.PositiveInfinity)
    val optimizer = new Optimizer(f = negativeLogLikelihood, l_u = (lowerBounds, upperBounds))

    // --- Solve and Report ---
    val (est_loss, est_params) = optimizer.solve(b0)

    println("\nEstimation Complete (Training Set):")
    for i <- 0 until p do println(s"Estimated phi_${i+1}:   ${est_params(i)}")
    println(s"Estimated sigma^2 (Q): ${est_params(p)}")
    println(s"Estimated R:         ${est_params(p + 1)}")

    // --- EVALUATION (Strictly on Test Data) ---
    println(s"\n--- Quality of Fit on TEST SET (Last 20%) for AR($p) ---")

    val b_kalman = est_params(0 until p)
    // Calculate intercept using Training Mean
    val delta_kalman = y_train.mean * (1 - b_kalman.sum)

    // 1-Step Ahead Prediction function
    // Uses 'y_hist' (full data) to access the lagged actuals [t-p ... t-1]
    def predict_kalman(t: Int, y_hist: VectorD): Double =
      if t < p then y_hist(t)
      else delta_kalman + (b_kalman dot y_hist.prior(p, t).reverse)
    end predict_kalman

    // Generate predictions for the Test Set range
    // We loop from 'train_size' to end of 'y'
    val yp_test = new VectorD(y_test.dim)
    for i <- 0 until y_test.dim do
      val t = train_size + i // Global index
      yp_test(i) = predict_kalman(t, y)
    end for

    class KalmanDiagnoser(n_params: Int, n_obs: Int)
      extends Diagnoser(dfm = n_params.toDouble, df = (n_obs - n_params).toDouble):
      val modName = s"Kalman-AR($n_params)"
    end KalmanDiagnoser

    // Diagnose compares Actual Test Data (y_test) vs Predicted Test Data (yp_test)
    val reporter = new KalmanDiagnoser(p, y_test.dim)
    val stats = reporter.diagnose(y_test, yp_test)

    println(s"Validation metrics for ${reporter.modName}:")
    for i <- stats.indices do
      if qoF_names(i) != "NA" then println(f"${qoF_names(i)}%10s = ${stats(i)}%12.6f")
    end for

  end for
end aR_KalmanEstimationTest_ILI

