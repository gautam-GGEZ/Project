from datetime import datetime
import flet as ft
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64, time, threading
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os
from flet import app
from Datapreprocessing import preprocessData
# Logging imports
from Logs import logInfo, logWarning, logError, logDebug, logData

# Email sending imports
import smtplib
from email.message import EmailMessage
import ssl

# Unified automationTools import
from automationTools import (
    logEvent,
    autoHyperparameterTuning,
    runCiPipeline,
    generateAiInsightSummary,
)

# Ensure DataTable and related classes are imported
from flet import DataTable, DataColumn, DataRow, DataCell

import json
with open("message.json", "r") as msg_file:
    allMessages = json.load(msg_file)

# --- Language selection ---
selectedLanguage = "en"
messages = allMessages


plotViews = {}  # Global dictionary for plots

# --- Email sending function using Gmail SMTP and attachment ---
def sendEmailWithAttachment(recipient, subject, body, csvContent):
    try:
        sender_email = "productforecastingapplication@gmail.com"
        app_password = "reclihabvciyshlk"
        msg = EmailMessage()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.set_content(body)

        # Attach the forecast CSV directly from memory
        msg.add_attachment(csvContent.encode('utf-8'),
                           maintype='application',
                           subtype='octet-stream',
                           filename="forecast_output.csv")

        context = ssl._create_unverified_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)

        return True
    except Exception as e:
        print(f"Email send failed: {e}")
        return False

def animateUpload(btn, page):
    for _ in range(2):
        for icon in ["🧼", "🧽", "🌀", "✅"]:
            btn.text = icon
            page.update()
            time.sleep(0.2)

def animateForecast(btn, page):
    for _ in range(3):
        for icon in ["📈", "📊", "📉", "Forecast ✅"]:
            btn.text = icon
            page.update()
            time.sleep(0.2)

def detectFrequency(dates):
    diffs = pd.Series(dates).sort_values().diff().dt.days.dropna()
    if diffs.empty:
        return "MS", "Monthly"
    avg_diff = diffs.mode().iloc[0]
    if avg_diff <= 8:
        return "W", "Weekly"
    elif avg_diff <= 15:
        return "SM", "Semi-Monthly"
    else:
        return "MS", "Monthly"

def closeDialog(e, page):
    global dlg
    dlg.open = False
    page.update()

def showPopup(page, title, msg, success=True):
    global dlg
    dlg = ft.AlertDialog(
        title=ft.Text(title),
        content=ft.Text(msg),
        actions=[ft.TextButton("OK", on_click=lambda e: closeDialog(e, page))],
        actions_alignment=ft.MainAxisAlignment.END
    )
    dlg.open = True
    page.dialog = dlg
    page.update()

def main(page: ft.Page):
    global dlg
    dlg = None  # Global variable for dialog
    page.title = "Sales Forecasting PRO"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.scroll = "none"
    page.padding = 0
    page.vertical_alignment = ft.MainAxisAlignment.START

    # messages is now just allMessages, no switching

    def getGradient():
        return ft.LinearGradient(
            begin=ft.alignment.top_left,
            end=ft.alignment.bottom_right,
            colors=["#1e1e1e", "#2a2a2a"] if page.theme_mode == ft.ThemeMode.DARK else ["#f0f4f8", "#dbe9f4"]
        )

    def showPlot(e):
        # Use label as key, fallback to a helpful message if not found
        plotForecast.content = plotViews.get(e.control.text, ft.Text("No chart found for this tab."))
        page.update()

    global forecastTabBtns
    forecastTabBtns = ft.Row([
        ft.TextButton("Forecasted Sales", on_click=showPlot),
        ft.TextButton("Forecasted Profit", on_click=showPlot),
        ft.TextButton("Top Products", on_click=showPlot),
        ft.TextButton("Segment Sales", on_click=showPlot),
    ])

    filePicker = ft.FilePicker()
    page.overlay.append(filePicker)

    forecastSummary = ft.Text("", size=16, italic=True)
    # AI Forecasted Insight Summary dynamic text elements with bold (color set dynamically later)
    aiSummaryText1 = ft.Text("", size=14, weight="bold")
    aiSummaryText2 = ft.Text("", size=14, weight="bold")
    aiSummaryText3 = ft.Text("", size=14, weight="bold")
    aiSummaryText4 = ft.Text("", size=14, weight="bold")
    aiSummaryText5 = ft.Text("", size=14, weight="bold")
    aiSummaryText6 = ft.Text("", size=14, weight="bold")
    aiSummaryText7 = ft.Text("", size=14, weight="bold")
    aiSummaryText8 = ft.Text("", size=14, weight="bold")
    aiSummaryText9 = ft.Text("", size=14, weight="bold")
    resultMetrics = ft.Text("", size=16)
    fileInfo = ft.Text("No file uploaded yet.")
    freqInfo = ft.Text("", size=14, italic=True, color=ft.Colors.BLUE)
    logOutput = ft.Text("", size=12, selectable=True, max_lines=10)
    automationOutputText = ft.Text("⚙️ Automation Logs will appear here...", size=14, color=ft.Colors.DEEP_PURPLE, selectable=True)
    emailHistoryText = ft.Text("📬 Email History:\n", size=13, color=ft.Colors.BLUE_900, selectable=True)
    plotForecast = ft.Container(padding=2, height=750, expand=True, alignment=ft.alignment.center)
    def toggleFullscreen():
        # Toggle between fixed height and fullscreen (None)
        plotForecast.height = None if plotForecast.height else 650
        page.update()

    # uploadBtn = ft.ElevatedButton("Upload File", width=200)
    forecastButton = ft.ElevatedButton("Run Forecast", icon=ft.Icons.SHOW_CHART, width=200)
    downloadCsvBtn = ft.ElevatedButton("⬇ Download CSV", visible=False)
    downloadExcelBtn = ft.ElevatedButton("⬇ Download Excel", visible=False)
    # --- Email Forecast UI ---
    emailInput = ft.TextField(label="Enter Email ID", width=250)
    sendEmailBtn = ft.ElevatedButton("📧 Send Forecast to Email", width=250)

    def sendForecastEmail(e):
        email = emailInput.value.strip()
        if not email or "@" not in email or "." not in email.split("@")[-1]:
            showPopup(page, "Invalid Email", messages.get("InvalidEmailFormat", "⚠️ Please enter a valid email address."), success=False)
            return
        sendEmailBtn.text = "Sending..."
        page.update()
        # Use the new email sending function with attachment
        emailAddress = email
        emailSubject = "Your Forecast Report"
        success = sendEmailWithAttachment(
            emailAddress,
            emailSubject,
            "The forecast output is ready. Please find the attached CSV file.",
            forecastCsv
        )
        if success:
            logInfo(f"Email sent to {emailAddress}")
            showPopup(page, "Email Sent", messages.get("EmailSent", f"✅ Email sent to {emailAddress}."), success=True)
            sendEmailBtn.text = "📧 Send Forecast to Email"
            emailHistoryText.value += f"• Sent to {emailAddress} at {datetime.now().strftime('%H:%M:%S')}\n"
            # --- Email log block ---
            logsFolder = "logs"
            os.makedirs(logsFolder, exist_ok=True)

            with open(os.path.join(logsFolder, "email_log.txt"), "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Email sent to {emailAddress} with subject '{emailSubject}'\n")
            # --- End email log block ---
            page.update()
        else:
            showPopup(page, "Email Error", messages.get("EmailFailed", f"❌ Failed to send email to {emailAddress}."), success=False)
            sendEmailBtn.text = "📧 Send Forecast to Email"
            emailHistoryText.value += f"• Failed to send to {emailAddress}\n"
            page.update()

    sendEmailBtn.on_click = sendForecastEmail

    loadingIndicator = ft.ProgressRing(visible=False, color="blue", scale=1.2)

    dateDropdown = ft.Dropdown(label="Date Column", width=200)
    salesDropdown = ft.Dropdown(label="Sales Column", width=200)
    modelDropdown = ft.Dropdown(
        label="Model",
        options=[
            ft.dropdown.Option("XGBoost"),
            ft.dropdown.Option("ARIMA"),
            ft.dropdown.Option("Compare Both")
        ],
        value="Compare Both", width=200
    )
    monthsSlider = ft.Slider(min=1, max=36, divisions=35, value=12, label="{value} periods", width=200)

    data = None
    forecastCsv = ""
    forecastExcel = None

    def showSnack(msg):
        page.snack_bar = ft.SnackBar(content=ft.Text(msg), duration=900)
        page.snack_bar.open = True
        page.update()

    # --- Language Dropdown ---
    # languageDropdown removed

    def toggleDark(e):
        page.theme_mode = ft.ThemeMode.DARK if e.control.value else ft.ThemeMode.LIGHT
        background_container.gradient = getGradient()
        page.update()


    # --- Smart Insights Panel ---
    highestMonthText = ft.Text("📌 Highest Sales Month:", size=15, weight="medium", text_align="center")
    lowestQuarterText = ft.Text("📉 Lowest Profit Quarter:", size=15, weight="medium", text_align="center")
    trendText = ft.Text("📈 Trend:", size=15, weight="medium", text_align="center")
    topRegionText = ft.Text("🛍️ Top Performing Region:", size=15, weight="medium", text_align="center")
    mostProfitableProductText = ft.Text("💰 Most Profitable Product:", size=15, weight="medium", text_align="center")
    leastSellingProductText = ft.Text("📉 Least Selling Product:", size=15, weight="medium", text_align="center")

    def updateSmartInsights():
        if data is None or data.empty:
            highestMonthText.value = "📌 Highest Sales Month: N/A"
            lowestQuarterText.value = "📉 Lowest Profit Quarter: N/A"
            trendText.value = "📈 Trend: N/A"
            topRegionText.value = "🛍️ Top Performing Region: N/A"
            mostProfitableProductText.value = "💰 Most Profitable Product: N/A"
            leastSellingProductText.value = "📉 Least Selling Product: N/A"
            return
        try:
            df = data.copy()
            df[dateDropdown.value] = pd.to_datetime(df[dateDropdown.value], errors="coerce")
            df.dropna(subset=[dateDropdown.value, salesDropdown.value], inplace=True)

            highestMonth = df.groupby(df[dateDropdown.value].dt.to_period("M"))[salesDropdown.value].sum().idxmax()
            lowestProfitQ = df.groupby(df[dateDropdown.value].dt.to_period("Q"))["Profit"].sum().idxmin()
            topRegion = df.groupby("Region")[salesDropdown.value].sum().idxmax()
            salesTrend = "📈 Increasing" if df.sort_values(dateDropdown.value)[salesDropdown.value].diff().mean() > 0 else "📉 Decreasing"

            # New smart insights
            mostProfitable = df.groupby("Product Name")["Profit"].sum().idxmax()
            leastSelling = df.groupby("Product Name")["Sales"].sum().idxmin()

            highestMonthText.value = f"📌 Highest Sales Month: {highestMonth}"
            lowestQuarterText.value = f"📉 Lowest Profit Quarter: {lowestProfitQ}"
            trendText.value = f"📈 Trend: {salesTrend}"
            topRegionText.value = f"🛍️ Top Performing Region: {topRegion}"
            mostProfitableProductText.value = f"💰 Most Profitable Product: {mostProfitable}"
            leastSellingProductText.value = f"📉 Least Selling Product: {leastSelling}"
            logInfo("Smart insights updated successfully.")
        except Exception as e:
            highestMonthText.value = "❌ Failed to generate insights"
            lowestQuarterText.value = ""
            trendText.value = ""
            topRegionText.value = f"{e}"
            mostProfitableProductText.value = "💰 Most Profitable Product: N/A"
            leastSellingProductText.value = "📉 Least Selling Product: N/A"

    def loadFile(e):
        nonlocal data
        loadingIndicator.visible = True
        # uploadBtn.disabled = True
        # uploadBtn.text = "📤 Uploading..."
        page.update()
        # threading.Thread(target=lambda: animateUpload(uploadBtn, page)).start()
        try:
            file = filePicker.result.files[0]
            if file.path.endswith(".csv"):
                for enc in ["utf-8", "ISO-8859-1", "cp1252"]:
                    try:
                        data = pd.read_csv(file.path, encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                data = pd.read_excel(file.path)

            # Log file upload and shape before cleaning
            logInfo(f"File uploaded: {file.name}")
            logInfo(f"File shape before cleaning: {data.shape}")

            fileInfo.value = messages.get("PreprocessingStarted", "🧹 Cleaning data...")
            page.update()
            time.sleep(0.7)
            data, log = preprocessData(data, column=salesDropdown.value if salesDropdown.value else "Sales")
            logOutput.value = "🧹 Data Cleaning Log:\n" + "\n".join(log)

            # Log after cleaning
            logInfo("Data cleaning complete.")
            logData("Cleaning Log", log)
            logInfo(f"File shape after cleaning: {data.shape}")

            cols = data.columns.tolist()
            dateDropdown.options = [ft.dropdown.Option(c) for c in cols]
            salesDropdown.options = [ft.dropdown.Option(c) for c in cols]
            # groupbyDropdown.options = [ft.dropdown.Option(c) for c in data.columns if data[c].dtype == "object"]
            dateDropdown.value = next((c for c in cols if "date" in c.lower()), cols[0])
            salesDropdown.value = next((c for c in cols if "sale" in c.lower() or "amount" in c.lower()), cols[-1])

            freqCode, freqLabel = detectFrequency(pd.to_datetime(data[dateDropdown.value], errors="coerce"))
            freqInfo.value = f"📅 Detected frequency: {freqLabel}"
            fileInfo.value = messages.get("DataLoadSuccess", f"✅ Loaded & cleaned: {file.name}")
            showSnack("✅ File uploaded and cleaned!")
        except Exception as ex:
            fileInfo.value = f"{messages.get('DataLoadError', '❌ Error loading data.')}: {ex}"
            logError(f"Exception during file loading: {ex}")

        # Update smart insights after loading file
        updateSmartInsights()
        # Log successful upload and processing
        logEvent("File successfully uploaded and processed.")
        # uploadBtn.disabled = False
        # uploadBtn.text = "📤 Upload File"
        loadingIndicator.visible = False
        page.update()

    def forecast(e=None):
        nonlocal forecastCsv, forecastExcel
        if data is None:
            fileInfo.value = messages.get("InputValidationError", "⚠️ Upload a file first.")
            return
        loadingIndicator.visible = True
        forecastButton.disabled = True
        forecastButton.text = "⏳ Forecasting..."
        page.update()
        threading.Thread(target=lambda: animateForecast(forecastButton, page)).start()
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            dcol, scol = dateDropdown.value, salesDropdown.value
            months = int(monthsSlider.value)
            modelType = modelDropdown.value

            # Log forecasting start
            logInfo(messages.get("ForecastGenerationStarted", "Forecasting started..."))
            logInfo(f"Selected model: {modelType}, Forecast horizon: {months} periods")

            # Run auto hyperparameter tuning
            try:
                autoHyperparameterTuning(
                    data[[dateDropdown.value, salesDropdown.value]].dropna(),
                    date_column=dateDropdown.value,
                    target_column=salesDropdown.value
                )
            except Exception as tuningEx:
                logWarning(f"Auto hyperparameter tuning failed: {tuningEx}")

            # Aggregate all columns dynamically, not just sales or profit
            df = data.dropna(subset=[dcol, scol])
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
            freqCode, freqLabel = detectFrequency(df[dcol])
            df = df.groupby(pd.Grouper(key=dcol, freq=freqCode)).sum()
            df = df[[scol]]
            series = df[scol].astype(float)

            train = series[:-months]
            test = series[-months:]
            forecastIndex = pd.date_range(series.index[-1], periods=months+1, freq=freqCode)[1:]

            dfFeat = pd.DataFrame({'y': train})
            for lag in range(1, 13):
                dfFeat[f"lag_{lag}"] = dfFeat['y'].shift(lag)
            dfFeat.dropna(inplace=True)
            X = dfFeat.drop("y", axis=1).values
            y = dfFeat['y'].values

            modelDir = os.path.join(os.path.dirname(__file__), "DevelopedModels")
            xgb = joblib.load(os.path.join(modelDir, "xgboost_model.pkl"))
            arima = joblib.load(os.path.join(modelDir, "arima_model.pkl"))

            predXgb = []
            history = list(train[-12:].values)
            for _ in range(months):
                yhat = xgb.predict(np.array(history[-12:]).reshape(1, -1))[0]
                predXgb.append(yhat)
                history.append(yhat)
            predXgb = pd.Series(predXgb, index=forecastIndex)

            predArima = pd.Series(arima.forecast(steps=months).values, index=forecastIndex)

            maeXgb = mean_absolute_error(test, predXgb)
            maeArima = mean_absolute_error(test, predArima)
            rmseXgb = np.sqrt(mean_squared_error(test, predXgb))
            rmseArima = np.sqrt(mean_squared_error(test, predArima))

            resultMetrics.value = (
                f"✅ MAE (XGBoost): {maeXgb:.2f} | RMSE: {rmseXgb:.2f}\n"
                f"✅ MAE (ARIMA): {maeArima:.2f} | RMSE: {rmseArima:.2f}"
            )

            # Log metrics
            logInfo(f"MAE (XGBoost): {maeXgb:.2f}, RMSE: {rmseXgb:.2f}")
            logInfo(f"MAE (ARIMA): {maeArima:.2f}, RMSE: {rmseArima:.2f}")

            # Forecasting Profit
            profitDf2 = data.dropna(subset=[dcol, "Profit"])
            profitDf2[dcol] = pd.to_datetime(profitDf2[dcol], errors="coerce")
            profitDf2 = profitDf2.groupby(pd.Grouper(key=dcol, freq=freqCode)).sum()
            profitDf2 = profitDf2[["Profit"]]
            profitSeries = profitDf2["Profit"].astype(float)

            trainProfit = profitSeries[:-months]
            testProfit = profitSeries[-months:]
            forecastIndexProfit = pd.date_range(profitSeries.index[-1], periods=months+1, freq=freqCode)[1:]

            dfFeatP = pd.DataFrame({'y': trainProfit})
            for lag in range(1, 13):
                dfFeatP[f"lag_{lag}"] = dfFeatP['y'].shift(lag)
            dfFeatP.dropna(inplace=True)
            Xp = dfFeatP.drop("y", axis=1).values
            yp = dfFeatP['y'].values

            predProfitXgb = []
            historyP = list(trainProfit[-12:].values)
            for _ in range(months):
                yhatP = xgb.predict(np.array(historyP[-12:]).reshape(1, -1))[0]
                predProfitXgb.append(yhatP)
                historyP.append(yhatP)
            predProfitXgb = pd.Series(predProfitXgb, index=forecastIndexProfit)

            if modelType != "XGBoost":
                predProfitArima = pd.Series(arima.forecast(steps=months).values, index=forecastIndexProfit)

            forecastDf = pd.DataFrame({
                "Date": forecastIndex,
                "XGBoost Forecast": predXgb,
                "ARIMA Forecast": predArima,
                "XGBoost Profit Forecast": predProfitXgb,
                "ARIMA Profit Forecast": predProfitArima if modelType != "XGBoost" else [None]*months
            })
            forecastCsv = forecastDf.to_csv(index=False)
            forecastExcel = forecastDf

            # Plots
            def makeImage(fig):
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                plt.close(fig)
                buf.seek(0)
                return ft.Image(src_base64=base64.b64encode(buf.read()).decode(), expand=True)

            # Forecast plot (show only selected model unless comparing both)
            fig1 = plt.figure(figsize=(12, 4))
            plt.plot(train, label="Train")
            plt.plot(test, label="Test")

            if modelType == "XGBoost":
                plt.plot(predXgb, "--", label="XGBoost")
            elif modelType == "ARIMA":
                plt.plot(predArima, ":", label="ARIMA")
            else:  # Compare Both
                plt.plot(predXgb, "--", label="XGBoost")
                plt.plot(predArima, ":", label="ARIMA")

            plt.title("Sales Forecast")
            plt.legend()
            forecastImage = makeImage(fig1)


            # Top 10 Products by Sales (with forecast if possible)
            figTop = plt.figure(figsize=(12, 4))
            if "Product Name" in data.columns:
                # Try to add forecasted sales by product for the forecast period
                data_cp = data.copy()
                data_cp[dcol] = pd.to_datetime(data_cp[dcol], errors="coerce")
                # Aggregate historical sales by product
                histTop = data_cp.groupby("Product Name")[scol].sum()
                # Try to estimate forecasted sales by product for the forecast period
                # Only possible if Product Name is available for each forecasted period (not typical for time series)
                # So: Mark with a note
                # Note: Top Products is based on total historical sales, not forecasted
                topProducts = histTop.sort_values(ascending=False).head(10)
                ax = topProducts.plot(kind="bar", color="skyblue")
                plt.title("Top 10 Products by Sales")
                plt.ylabel("Sales")
                plt.xticks(rotation=30, ha="center", fontsize=10, wrap=True)
                ax.set_xticklabels([label.get_text()[:15] + '...' if len(label.get_text()) > 18 else label.get_text() for label in ax.get_xticklabels()])
                # Note: Top Products is based on total historical sales, not forecasted
                # plt.text(0.99, 0.95, "Note: Top Products is based on total historical sales, not forecasted", ha="right", va="top", transform=plt.gca().transAxes, fontsize=9, color="gray")
            else:
                # Note: Top Products is based on total historical sales, not forecasted
                plt.text(0.5, 0.5, "Note: Top Products is based on total historical sales, not forecasted", ha="center", va="center", fontsize=14)
                plt.title("Top 10 Products by Sales")
            topImage = makeImage(figTop)

            # Sales by Segment or Region (use recent/forecast-relevant data)
            figSeg = plt.figure(figsize=(12, 4))
            segSales = data.copy()
            segSales[dcol] = pd.to_datetime(segSales[dcol], errors="coerce")
            segSales = segSales[segSales[dcol] >= df.index[-months]]  # include only recent data
            segData = segSales.groupby(["Segment", "Region"])[scol].sum().unstack().fillna(0)
            segData.plot(kind="bar", stacked=False, ax=plt.gca())
            plt.title("Sales by Segment and Region")
            plt.ylabel("Sales")
            plt.xticks(rotation=0)
            segmentImage = makeImage(figSeg)

            figProfitForecast = plt.figure(figsize=(12, 4))
            plt.plot(trainProfit, label="Train Profit", color="green")
            plt.plot(testProfit, label="Test Profit", color="orange")

            if modelType == "XGBoost":
                plt.plot(predProfitXgb, "--", label="XGBoost Profit Forecast", color="red")
            elif modelType == "ARIMA":
                plt.plot(predProfitArima, ":", label="ARIMA Profit Forecast", color="purple")
            else:  # Compare Both
                plt.plot(predProfitXgb, "--", label="XGBoost Profit Forecast", color="red")
                plt.plot(predProfitArima, ":", label="ARIMA Profit Forecast", color="purple")

            plt.title("Forecasted Profit Over Time")
            plt.legend()
            profitForecastImage = makeImage(figProfitForecast)

            plotViews["Forecasted Sales"] = forecastImage
            plotViews["Forecasted Profit"] = profitForecastImage
            plotViews["Top Products"] = topImage
            plotViews["Segment Sales"] = segmentImage
            # Ensure the default tab displays the correct chart
            plotForecast.content = plotViews["Forecasted Sales"]

            trend = "increase 📈" if predXgb.mean() > train[-months:].mean() else "decrease 📉"
            forecastSummary.value = f"🔎 Based on the forecast, sales are expected to {trend}."


            # Publish forecast data to dashboard and generate summary

            # --- Automation Tools Integration ---
            log_block = []

            runCiPipeline()
            log_block.append("✅ CI/CD pipeline executed.")


            # Removed simulated report email sending (no attachment)

            log_block.append("📊 Dashboard preview simulated (feature under development)")

            summary_text = f"XGBoost MAE: {maeXgb:.2f}, XGBoost RMSE: {rmseXgb:.2f}"
            logEvent(summary_text)
            log_block.append(f"🧠 Summary generated: {summary_text}")

            automationOutputText.value = "\n".join(log_block)
            # AI Insight Summary update
            summaryText = generateAiInsightSummary(forecastDf)
            # Parse summaryText into lines and update dynamic AI summary texts (values only, no emojis/labels)
            aiLines = summaryText.splitlines()
            # Improved extractValue function for sanitizing and stripping extra characters
            def extractValue(line):
                return line.strip("•-#:\n\t ").replace("**", "").strip() if line else "N/A"

            aiSummaryText1.value = f"{extractValue(aiLines[1]) if len(aiLines) > 1 and extractValue(aiLines[1]) else 'N/A'}"
            aiSummaryText2.value = f"{extractValue(aiLines[2]) if len(aiLines) > 2 and extractValue(aiLines[2]) else 'N/A'}"
            aiSummaryText3.value = f"{extractValue(aiLines[3]) if len(aiLines) > 3 and extractValue(aiLines[3]) else 'N/A'}"
            aiSummaryText4.value = f"{extractValue(aiLines[4]) if len(aiLines) > 4 and extractValue(aiLines[4]) else 'N/A'}"
            aiSummaryText5.value = f"{extractValue(aiLines[5]) if len(aiLines) > 5 and extractValue(aiLines[5]) else 'N/A'}"
            aiSummaryText6.value = f"{extractValue(aiLines[6]) if len(aiLines) > 6 and extractValue(aiLines[6]) else 'N/A'}"
            aiSummaryText7.value = f"{extractValue(aiLines[1]) if len(aiLines) > 1 and extractValue(aiLines[1]) else ''}"
            aiSummaryText8.value = extractValue(aiLines[8]) if len(aiLines) > 8 and extractValue(aiLines[8]) else ""
            aiSummaryText9.value = extractValue(aiLines[9]) if len(aiLines) > 9 and extractValue(aiLines[9]) else ""
            # forecastSummary.value += f"\n\n🔍 Summary:\n{summaryText}"
            downloadCsvBtn.visible = True
            downloadExcelBtn.visible = True
            showSnack(messages.get("ForecastGenerationSuccess", "✅ Forecast complete!"))
        except Exception as ex:
            fileInfo.value = f"{messages.get('ForecastGenerationFailure', '❌ Forecast failed.')}: {ex}"
            logError(f"Exception during forecasting: {ex}")

        forecastButton.disabled = False
        forecastButton.text = "📈 Forecast"
        loadingIndicator.visible = False
        page.update()

    # Load default dataset
    def loadDefaultDataset():
        nonlocal data
        try:
            path = os.path.join(os.path.dirname(__file__), "Dataset", "Dataset.csv")
            for enc in ["utf-8", "ISO-8859-1", "cp1252"]:
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            data, log = preprocessData(df, column="Sales")
            logOutput.value = "🧹 Data Cleaning Log:\n" + "\n".join(log)
            dateDropdown.options = [ft.dropdown.Option(c) for c in data.columns]
            salesDropdown.options = [ft.dropdown.Option(c) for c in data.columns]
            # groupbyDropdown.options = [ft.dropdown.Option(c) for c in data.columns if data[c].dtype == "object"]
            dateDropdown.value = next((c for c in data.columns if "date" in c.lower()), data.columns[0])
            salesDropdown.value = next((c for c in data.columns if "sale" in c.lower() or "amount" in c.lower()), data.columns[-1])
            freqCode, freqLabel = detectFrequency(pd.to_datetime(data[dateDropdown.value], errors="coerce"))
            freqInfo.value = f"📅 Detected frequency: {freqLabel}"
            fileInfo.value = "✅ Default dataset loaded"
            plotForecast.content = None  # clear any previous chart preview
            # Update smart insights after loading default dataset
            updateSmartInsights()
            page.update()
            forecast(None)
        except Exception as e:
            fileInfo.value = f"❌ Failed to load default dataset: {e}"

    # uploadBtn.on_click = lambda _: filePicker.pick_files(allow_multiple=False, allowed_extensions=["csv", "xlsx"])
    filePicker.on_result = loadFile
    forecastButton.on_click = forecast

    def downloadCsv(e):
        if forecastCsv:
            with open("forecast_output.csv", "w") as f:
                f.write(forecastCsv)
            os.system("open forecast_output.csv")

    def downloadExcel(e):
        if forecastExcel is not None:
            forecastExcel.to_excel("forecast_output.xlsx", index=False)
            os.system("open forecast_output.xlsx")

    downloadCsvBtn.on_click = downloadCsv
    downloadExcelBtn.on_click = downloadExcel

    panel_list = [
        ft.ExpansionPanel(
            header=ft.Container(
                content=ft.Text("Data Settings", size=16, weight="bold"),
                padding=ft.padding.only(left=8, top=5, bottom=5)
            ),
            content=ft.Container(
                padding=ft.padding.only(left=10, right=10),
                content=ft.Column([
                    ft.Row([
                        ft.Text("Date Column", size=13, weight="medium"),
                        ft.IconButton(icon=ft.Icons.INFO_OUTLINE, tooltip="Select the date field in your data.")
                    ]),
                    ft.Row([dateDropdown], alignment="center"),
                    ft.Row([
                        ft.Text("Sales Column", size=13, weight="medium"),
                        ft.IconButton(icon=ft.Icons.INFO_OUTLINE, tooltip="Select the numeric field that holds sales data.")
                    ]),
                    ft.Row([salesDropdown], alignment="center"),
                    ft.Container(height=10)  # spacer to prevent visual collision at bottom
                ], spacing=16)
            ),
            expanded=True
        ),
        ft.ExpansionPanel(
            header=ft.Container(
                content=ft.Text("Model Configuration", size=16, weight="bold"),
                padding=ft.padding.only(left=8, top=5, bottom=5)
            ),
            content=ft.Container(
                padding=ft.padding.only(left=10, right=10),
                content=ft.Column([
                    ft.Row([
                        ft.Text("Forecasting Model", size=13, weight="medium"),
                        ft.IconButton(icon=ft.Icons.INFO_OUTLINE, tooltip="Choose XGBoost, ARIMA or compare both.")
                    ]),
                    ft.Row([modelDropdown], alignment="center"),
                    ft.Row([
                        ft.Text("Periods to Forecast", size=13, weight="medium"),
                        ft.IconButton(icon=ft.Icons.INFO_OUTLINE, tooltip="How many future time periods to forecast?")
                    ]),
                    ft.Row([monthsSlider], alignment="center")
                ], spacing=16)
            ),
            expanded=True
        ),
        ft.ExpansionPanel(
            header=ft.Container(
                content=ft.Text("Run & Upload", size=16, weight="bold"),
                padding=ft.padding.only(left=8, top=5, bottom=5)
            ),
            content=ft.Container(
                padding=ft.padding.only(left=10, right=10),
                content=ft.Column([
                    ft.Row([forecastButton], alignment="center"),
                    # ft.Row([upload_btn], alignment="center"),
                    ft.Container(height=10)
                ], spacing=16)
            ),
            expanded=True
        )
    ]

    sidebar = ft.Container(
        width=400,
        bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.BLUE_GREY),
        padding=10,
        content=ft.Column([
            ft.Text("📊 Controls", size=20, weight="bold", text_align="start"),
            ft.Container(
                padding=ft.padding.only(left=10, right=5),
                content=ft.Container(
                    content=ft.ExpansionPanelList(controls=panel_list),
                    border_radius=10,
                    padding=10,
                    bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.BLUE_GREY)
                )
            ),
            ft.Divider(),
            ft.Text("📌 Summary", weight="bold"),
            ft.Container(
                content=ft.Column([
                    ft.Text("📁 File:", size=13, weight="medium"),
                    fileInfo,
                    ft.Text("⏳ Frequency:", size=13, weight="medium"),
                    freqInfo,
                    ft.Text("📌 Cleaning Log", weight="bold"),
                    logOutput,
                ]),
                bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.BLUE_GREY),
                padding=10,
                border_radius=8
            )
        ], scroll=ft.ScrollMode.AUTO)
    )

    background_container = ft.Container(
        expand=True,
        gradient=getGradient(),
        content=ft.Column([
            ft.Container(
                ft.Row([
                    ft.Text("Sales Forecasting Dashboard", size=22, weight="bold"),
                    ft.Row([
                        ft.Switch(label="🌙 Dark Mode", value=False, on_change=toggleDark)
                    ], spacing=10)
                ], alignment="spaceBetween"),
                padding=15,
                bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.BLUE_GREY)
            ),
            ft.Text("📦 Welcome to the Store Sales Forecasting Application!", size=14, italic=True),
            # Removed smart insights container from here
            ft.Row([
                sidebar,
                ft.Container(expand=True, content=ft.Column([
                    ft.Row([loadingIndicator], alignment="center"),
                    # Project info/status/tip row
                    ft.Container(
                        content=ft.Row([
                            ft.Text("🧠 Project: Store Sales Forecasting", size=14, weight="bold"),
                            ft.Text("🔄 Status: Ready", size=13, italic=True),
                            ft.Text("💡 Tip: Use 'Compare Both' to validate model performance!", size=13, italic=True),
                        ], alignment="spaceAround"),
                        padding=10
                    ),
                    resultMetrics,
                    forecastSummary,
                    ft.Container(
                        content=ft.Column([
                            ft.Text(
                                "🧠 Smart Insights Panel (OLD Data)",
                                size=18,
                                weight="bold",
                                color=ft.Colors.BLUE_900,
                                text_align="center"
                            ),
                            ft.Divider(thickness=1),
                            ft.Row([
                                ft.Column([
                                    highestMonthText,
                                    lowestQuarterText,
                                ], spacing=10, expand=True, alignment="center"),
                                ft.Column([
                                    trendText,
                                    topRegionText,
                                ], spacing=10, expand=True, alignment="center"),
                                ft.Column([
                                    mostProfitableProductText,
                                    leastSellingProductText,
                                ], spacing=10, expand=True, alignment="center")
                            ], spacing=40)
                        ], spacing=10, horizontal_alignment="center"),
                        padding=20,
                        bgcolor=ft.Colors.with_opacity(0.06, ft.Colors.BLUE_GREY),
                        border_radius=12,
                        margin=ft.margin.only(top=10, bottom=10)
                    ),
                    # Insert AI Insight Summary panel here (updated styling)
                    ft.Container(
                        content=ft.Column([
                            ft.Text(
                                "🧠 Forecasted Insight Summary",
                                size=18,
                                weight="bold",
                                color=ft.Colors.GREEN_900,
                                text_align="center"
                            ),
                            ft.Divider(thickness=1),
                            ft.Row([
                                ft.Column([aiSummaryText2], alignment="center", expand=True),
                                ft.Column([aiSummaryText3], alignment="center", expand=True),
                                ft.Column([aiSummaryText4], alignment="center", expand=True),
                            ], alignment="spaceEvenly"),
                            ft.Row([
                                ft.Column([aiSummaryText5], alignment="center", expand=True),
                                ft.Column([aiSummaryText6], alignment="center", expand=True),
                                ft.Column([aiSummaryText7], alignment="center", expand=True),
                            ], alignment="spaceEvenly")
                        ], spacing=10, horizontal_alignment="center"),
                        padding=20,
                        bgcolor=ft.Colors.with_opacity(0.06, ft.Colors.GREEN),
                        border_radius=12,
                        margin=ft.margin.only(top=10, bottom=10)
                    ),
                    ft.Container(
                        content=ft.Column([

                            emailHistoryText
                        ]),
                        bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.BLUE),
                        padding=15,
                        border_radius=10,
                        margin=ft.margin.only(top=5, bottom=10)
                    ),
                    ft.Row([downloadCsvBtn, downloadExcelBtn], alignment="center"),
                    ft.Row([emailInput, sendEmailBtn], alignment="center"),
                    # Insert tab buttons in a row (fullscreen button removed)
                    ft.Row([forecastTabBtns], alignment="center"),
                    # Make plot_forecast scrollable and expandable for multi-plot comparison
                    ft.Row([
                        ft.Container(content=ft.Column([plotForecast], scroll=ft.ScrollMode.AUTO), expand=True)
                    ], alignment="center"),
                ], scroll=ft.ScrollMode.AUTO, expand=True, horizontal_alignment="center"))
            ], expand=True),
            ft.Container(
                ft.Row([
                    ft.Text("© 2025 Hochschule Emden-Leer | Sales Forecasting App", size=12, italic=True),
                    ft.Text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), size=12, italic=True)
                ], alignment="spaceBetween", expand=True),
                padding=15,
                alignment=ft.alignment.bottom_center,
                bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.BLUE_GREY)
            )
        ], expand=True, spacing=0, alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
    )

    loadDefaultDataset()
    page.add(background_container)

#
# --- AI Insight Summary Generator ---
def generateAiInsightSummary(df):
    if df.empty:
        return "No insights available due to empty forecast data."
    try:
        salesForecast = df["XGBoost Forecast"] if "XGBoost Forecast" in df.columns else df.iloc[:, 1]
        trend = "rising 📈" if salesForecast.iloc[-1] > salesForecast.iloc[0] else "declining 📉"
        maxForecast = salesForecast.max()
        minForecast = salesForecast.min()
        return (
            f"- Predicted sales show a {trend} trend.\n"
            f"- Highest forecasted value: {maxForecast:.2f}\n"
            f"- Lowest forecasted value: {minForecast:.2f}\n"
            f"- Forecast spans {len(salesForecast)} periods.\n"
            f"- Total forecasted revenue: {salesForecast.sum():.2f}\n"
            f"- Average forecasted sales per period: {salesForecast.mean():.2f}\n"
            f"- Standard deviation (volatility): {salesForecast.std():.2f}"
        )
    except Exception as e:
        return f"⚠️ Failed to generate insight summary: {e}"

if __name__ == "__main__":
    app(target=main)