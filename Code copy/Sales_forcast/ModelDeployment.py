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
from Datapreprocessing import preprocess_data
# Logging imports
from Logs import log_info, log_warning, log_error, log_debug, log_data

# Email sending imports
import smtplib
from email.message import EmailMessage
import ssl

# Unified automationTools import
from automationTools import (
    log_event,
    auto_hyperparameter_tuning,
    run_ci_pipeline,
)

# Ensure DataTable and related classes are imported
from flet import DataTable, DataColumn, DataRow, DataCell

plot_views = {}  # Global dictionary for plots

# --- Email sending function using Gmail SMTP and attachment ---
def send_email_with_attachment(recipient, subject, body, csv_content):
    try:
        sender_email = "productforecastingapplication@gmail.com"
        app_password = "reclihabvciyshlk"
        msg = EmailMessage()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.set_content(body)

        # Attach the forecast CSV directly from memory
        msg.add_attachment(csv_content.encode('utf-8'),
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

def animate_upload(btn, page):
    for _ in range(2):
        for icon in ["🧼", "🧽", "🌀", "✅"]:
            btn.text = icon
            page.update()
            time.sleep(0.2)

def animate_forecast(btn, page):
    for _ in range(3):
        for icon in ["📈", "📊", "📉", "Forecast ✅"]:
            btn.text = icon
            page.update()
            time.sleep(0.2)

def detect_frequency(dates):
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

def close_dialog(e, page):
    global dlg
    dlg.open = False
    page.update()

def show_popup(page, title, msg, success=True):
    global dlg
    dlg = ft.AlertDialog(
        title=ft.Text(title),
        content=ft.Text(msg),
        actions=[ft.TextButton("OK", on_click=lambda e: close_dialog(e, page))],
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

    def get_gradient():
        return ft.LinearGradient(
            begin=ft.alignment.top_left,
            end=ft.alignment.bottom_right,
            colors=["#1e1e1e", "#2a2a2a"] if page.theme_mode == ft.ThemeMode.DARK else ["#f0f4f8", "#dbe9f4"]
        )

    def show_plot(e):
        # Use label as key, fallback to a helpful message if not found
        plot_forecast.content = plot_views.get(e.control.text, ft.Text("No chart found for this tab."))
        page.update()

    global forecast_tab_btns
    forecast_tab_btns = ft.Row([
        ft.TextButton("Forecasted Sales", on_click=show_plot),
        ft.TextButton("Forecasted Profit", on_click=show_plot),
        ft.TextButton("Top Products", on_click=show_plot),
        ft.TextButton("Segment Sales", on_click=show_plot),
    ])

    file_picker = ft.FilePicker()
    page.overlay.append(file_picker)

    forecast_summary = ft.Text("", size=16, italic=True)
    # AI Forecasted Insight Summary dynamic text elements
    ai_summary_text_1 = ft.Text("", size=14)
    ai_summary_text_2 = ft.Text("", size=14)
    ai_summary_text_3 = ft.Text("", size=14)
    ai_summary_text_4 = ft.Text("", size=14)
    ai_summary_text_5 = ft.Text("", size=14)
    ai_summary_text_6 = ft.Text("", size=14)
    ai_summary_text_7 = ft.Text("", size=14)
    ai_summary_text_8 = ft.Text("", size=14)
    ai_summary_text_9 = ft.Text("", size=14)
    result_metrics = ft.Text("", size=16)
    file_info = ft.Text("No file uploaded yet.")
    freq_info = ft.Text("", size=14, italic=True, color=ft.Colors.BLUE)
    log_output = ft.Text("", size=12, selectable=True, max_lines=10)
    automation_output_text = ft.Text("⚙️ Automation Logs will appear here...", size=14, color=ft.Colors.DEEP_PURPLE, selectable=True)
    email_history_text = ft.Text("📬 Email History:\n", size=13, color=ft.Colors.BLUE_900, selectable=True)
    plot_forecast = ft.Container(padding=2, height=750, expand=True, alignment=ft.alignment.center)
    def toggle_fullscreen():
        # Toggle between fixed height and fullscreen (None)
        plot_forecast.height = None if plot_forecast.height else 650
        page.update()

    # upload_btn = ft.ElevatedButton("Upload File", width=200)
    forecast_btn = ft.ElevatedButton("Run Forecast", icon=ft.Icons.SHOW_CHART, width=200)
    download_csv_btn = ft.ElevatedButton("⬇ Download CSV", visible=False)
    download_excel_btn = ft.ElevatedButton("⬇ Download Excel", visible=False)
    # --- Email Forecast UI ---
    email_input = ft.TextField(label="Enter Email ID", width=250)
    send_email_btn = ft.ElevatedButton("📧 Send Forecast to Email", width=250)

    def send_forecast_email(e):
        email = email_input.value.strip()
        if not email or "@" not in email or "." not in email.split("@")[-1]:
            show_popup(page, "Invalid Email", "⚠️ Please enter a valid email address.", success=False)
            return
        send_email_btn.text = "Sending..."
        page.update()
        # Use the new email sending function with attachment
        email_address = email
        email_subject = "Your Forecast Report"
        success = send_email_with_attachment(
            email_address,
            email_subject,
            "The forecast output is ready. Please find the attached CSV file.",
            forecast_csv
        )
        if success:
            log_info(f"Email sent to {email_address}")
            show_popup(page, "Email Sent", f"✅ Email sent to {email_address}. Please check your inbox in a few minutes.", success=True)
            send_email_btn.text = "📧 Send Forecast to Email"
            email_history_text.value += f"• Sent to {email_address} at {datetime.now().strftime('%H:%M:%S')}\n"
            # --- Email log block ---
            logs_folder = "logs"
            os.makedirs(logs_folder, exist_ok=True)

            with open(os.path.join(logs_folder, "email_log.txt"), "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Email sent to {email_address} with subject '{email_subject}'\n")
            # --- End email log block ---
            page.update()
        else:
            show_popup(page, "Email Error", f"❌ Failed to send email to {email_address}.", success=False)
            send_email_btn.text = "📧 Send Forecast to Email"
            email_history_text.value += f"• Failed to send to {email_address}\n"
            page.update()

    send_email_btn.on_click = send_forecast_email

    loading_indicator = ft.ProgressRing(visible=False, color="blue", scale=1.2)

    date_dropdown = ft.Dropdown(label="Date Column", width=200)
    sales_dropdown = ft.Dropdown(label="Sales Column", width=200)
    model_dropdown = ft.Dropdown(
        label="Model",
        options=[
            ft.dropdown.Option("XGBoost"),
            ft.dropdown.Option("ARIMA"),
            ft.dropdown.Option("Compare Both")
        ],
        value="Compare Both", width=200
    )
    months_slider = ft.Slider(min=1, max=36, divisions=35, value=12, label="{value} periods", width=200)

    data = None
    forecast_csv = ""
    forecast_excel = None

    def show_snack(msg):
        page.snack_bar = ft.SnackBar(content=ft.Text(msg), duration=900)
        page.snack_bar.open = True
        page.update()

    def toggle_dark(e):
        page.theme_mode = ft.ThemeMode.DARK if e.control.value else ft.ThemeMode.LIGHT
        background_container.gradient = get_gradient()
        page.update()


    # --- Smart Insights Panel ---
    highest_month_text = ft.Text("📌 Highest Sales Month:", size=15, weight="medium", text_align="center")
    lowest_quarter_text = ft.Text("📉 Lowest Profit Quarter:", size=15, weight="medium", text_align="center")
    trend_text = ft.Text("📈 Trend:", size=15, weight="medium", text_align="center")
    top_region_text = ft.Text("🛍️ Top Performing Region:", size=15, weight="medium", text_align="center")
    most_profitable_product_text = ft.Text("💰 Most Profitable Product:", size=15, weight="medium", text_align="center")
    least_selling_product_text = ft.Text("📉 Least Selling Product:", size=15, weight="medium", text_align="center")

    def update_smart_insights():
        if data is None or data.empty:
            highest_month_text.value = "📌 Highest Sales Month: N/A"
            lowest_quarter_text.value = "📉 Lowest Profit Quarter: N/A"
            trend_text.value = "📈 Trend: N/A"
            top_region_text.value = "🛍️ Top Performing Region: N/A"
            most_profitable_product_text.value = "💰 Most Profitable Product: N/A"
            least_selling_product_text.value = "📉 Least Selling Product: N/A"
            return
        try:
            df = data.copy()
            df[date_dropdown.value] = pd.to_datetime(df[date_dropdown.value], errors="coerce")
            df.dropna(subset=[date_dropdown.value, sales_dropdown.value], inplace=True)

            highest_month = df.groupby(df[date_dropdown.value].dt.to_period("M"))[sales_dropdown.value].sum().idxmax()
            lowest_profit_q = df.groupby(df[date_dropdown.value].dt.to_period("Q"))["Profit"].sum().idxmin()
            top_region = df.groupby("Region")[sales_dropdown.value].sum().idxmax()
            sales_trend = "📈 Increasing" if df.sort_values(date_dropdown.value)[sales_dropdown.value].diff().mean() > 0 else "📉 Decreasing"

            # New smart insights
            most_profitable = df.groupby("Product Name")["Profit"].sum().idxmax()
            least_selling = df.groupby("Product Name")["Sales"].sum().idxmin()

            highest_month_text.value = f"📌 Highest Sales Month: {highest_month}"
            lowest_quarter_text.value = f"📉 Lowest Profit Quarter: {lowest_profit_q}"
            trend_text.value = f"📈 Trend: {sales_trend}"
            top_region_text.value = f"🛍️ Top Performing Region: {top_region}"
            most_profitable_product_text.value = f"💰 Most Profitable Product: {most_profitable}"
            least_selling_product_text.value = f"📉 Least Selling Product: {least_selling}"
            log_info("Smart insights updated successfully.")
        except Exception as e:
            highest_month_text.value = "❌ Failed to generate insights"
            lowest_quarter_text.value = ""
            trend_text.value = ""
            top_region_text.value = f"{e}"
            most_profitable_product_text.value = "💰 Most Profitable Product: N/A"
            least_selling_product_text.value = "📉 Least Selling Product: N/A"

    def load_file(e):
        nonlocal data
        loading_indicator.visible = True
        # upload_btn.disabled = True
        # upload_btn.text = "📤 Uploading..."
        page.update()
        # threading.Thread(target=lambda: animate_upload(upload_btn, page)).start()
        try:
            file = file_picker.result.files[0]
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
            log_info(f"File uploaded: {file.name}")
            log_info(f"File shape before cleaning: {data.shape}")

            file_info.value = "🧹 Cleaning data..."
            page.update()
            time.sleep(0.7)
            data, log = preprocess_data(data, column=sales_dropdown.value if sales_dropdown.value else "Sales")
            log_output.value = "🧹 Data Cleaning Log:\n" + "\n".join(log)

            # Log after cleaning
            log_info("Data cleaning complete.")
            log_data("Cleaning Log", log)
            log_info(f"File shape after cleaning: {data.shape}")

            cols = data.columns.tolist()
            date_dropdown.options = [ft.dropdown.Option(c) for c in cols]
            sales_dropdown.options = [ft.dropdown.Option(c) for c in cols]
            # groupby_dropdown.options = [ft.dropdown.Option(c) for c in data.columns if data[c].dtype == "object"]
            date_dropdown.value = next((c for c in cols if "date" in c.lower()), cols[0])
            sales_dropdown.value = next((c for c in cols if "sale" in c.lower() or "amount" in c.lower()), cols[-1])

            freq_code, freq_label = detect_frequency(pd.to_datetime(data[date_dropdown.value], errors="coerce"))
            freq_info.value = f"📅 Detected frequency: {freq_label}"
            file_info.value = f"✅ Loaded & cleaned: {file.name}"
            show_snack("✅ File uploaded and cleaned!")
        except Exception as ex:
            file_info.value = f"❌ Error: {ex}"
            log_error(f"Exception during file loading: {ex}")

        # Update smart insights after loading file
        update_smart_insights()
        # Log successful upload and processing
        log_event("File successfully uploaded and processed.")
        # upload_btn.disabled = False
        # upload_btn.text = "📤 Upload File"
        loading_indicator.visible = False
        page.update()

    def forecast(e=None):
        nonlocal forecast_csv, forecast_excel
        if data is None:
            file_info.value = "⚠️ Upload a file first."
            return
        loading_indicator.visible = True
        forecast_btn.disabled = True
        forecast_btn.text = "⏳ Forecasting..."
        page.update()
        threading.Thread(target=lambda: animate_forecast(forecast_btn, page)).start()
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            dcol, scol = date_dropdown.value, sales_dropdown.value
            months = int(months_slider.value)
            model_type = model_dropdown.value

            # Log forecasting start
            log_info("Forecasting started...")
            log_info(f"Selected model: {model_type}, Forecast horizon: {months} periods")

            # Run auto hyperparameter tuning
            try:
                auto_hyperparameter_tuning(
                    data[[date_dropdown.value, sales_dropdown.value]].dropna(),
                    date_column=date_dropdown.value,
                    target_column=sales_dropdown.value
                )
            except Exception as tuning_ex:
                log_warning(f"Auto hyperparameter tuning failed: {tuning_ex}")

            # Aggregate all columns dynamically, not just sales or profit
            df = data.dropna(subset=[dcol, scol])
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
            freq_code, freq_label = detect_frequency(df[dcol])
            df = df.groupby(pd.Grouper(key=dcol, freq=freq_code)).sum()
            df = df[[scol]]
            series = df[scol].astype(float)

            train = series[:-months]
            test = series[-months:]
            forecast_index = pd.date_range(series.index[-1], periods=months+1, freq=freq_code)[1:]

            df_feat = pd.DataFrame({'y': train})
            for lag in range(1, 13):
                df_feat[f"lag_{lag}"] = df_feat['y'].shift(lag)
            df_feat.dropna(inplace=True)
            X = df_feat.drop("y", axis=1).values
            y = df_feat['y'].values

            model_dir = os.path.join(os.path.dirname(__file__), "DevelopedModels")
            xgb = joblib.load(os.path.join(model_dir, "xgboost_model.pkl"))
            arima = joblib.load(os.path.join(model_dir, "arima_model.pkl"))

            pred_xgb = []
            history = list(train[-12:].values)
            for _ in range(months):
                yhat = xgb.predict(np.array(history[-12:]).reshape(1, -1))[0]
                pred_xgb.append(yhat)
                history.append(yhat)
            pred_xgb = pd.Series(pred_xgb, index=forecast_index)

            pred_arima = pd.Series(arima.forecast(steps=months).values, index=forecast_index)

            mae_xgb = mean_absolute_error(test, pred_xgb)
            mae_arima = mean_absolute_error(test, pred_arima)
            rmse_xgb = np.sqrt(mean_squared_error(test, pred_xgb))
            rmse_arima = np.sqrt(mean_squared_error(test, pred_arima))

            result_metrics.value = (
                f"✅ MAE (XGBoost): {mae_xgb:.2f} | RMSE: {rmse_xgb:.2f}\n"
                f"✅ MAE (ARIMA): {mae_arima:.2f} | RMSE: {rmse_arima:.2f}"
            )

            # Log metrics
            log_info(f"MAE (XGBoost): {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}")
            log_info(f"MAE (ARIMA): {mae_arima:.2f}, RMSE: {rmse_arima:.2f}")

            # Forecasting Profit
            profit_df2 = data.dropna(subset=[dcol, "Profit"])
            profit_df2[dcol] = pd.to_datetime(profit_df2[dcol], errors="coerce")
            profit_df2 = profit_df2.groupby(pd.Grouper(key=dcol, freq=freq_code)).sum()
            profit_df2 = profit_df2[["Profit"]]
            profit_series = profit_df2["Profit"].astype(float)

            train_profit = profit_series[:-months]
            test_profit = profit_series[-months:]
            forecast_index_profit = pd.date_range(profit_series.index[-1], periods=months+1, freq=freq_code)[1:]

            df_feat_p = pd.DataFrame({'y': train_profit})
            for lag in range(1, 13):
                df_feat_p[f"lag_{lag}"] = df_feat_p['y'].shift(lag)
            df_feat_p.dropna(inplace=True)
            X_p = df_feat_p.drop("y", axis=1).values
            y_p = df_feat_p['y'].values

            pred_profit_xgb = []
            history_p = list(train_profit[-12:].values)
            for _ in range(months):
                yhat_p = xgb.predict(np.array(history_p[-12:]).reshape(1, -1))[0]
                pred_profit_xgb.append(yhat_p)
                history_p.append(yhat_p)
            pred_profit_xgb = pd.Series(pred_profit_xgb, index=forecast_index_profit)

            if model_type != "XGBoost":
                pred_profit_arima = pd.Series(arima.forecast(steps=months).values, index=forecast_index_profit)

            forecast_df = pd.DataFrame({
                "Date": forecast_index,
                "XGBoost Forecast": pred_xgb,
                "ARIMA Forecast": pred_arima,
                "XGBoost Profit Forecast": pred_profit_xgb,
                "ARIMA Profit Forecast": pred_profit_arima if model_type != "XGBoost" else [None]*months
            })
            forecast_csv = forecast_df.to_csv(index=False)
            forecast_excel = forecast_df

            # Plots
            def make_image(fig):
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                plt.close(fig)
                buf.seek(0)
                return ft.Image(src_base64=base64.b64encode(buf.read()).decode(), expand=True)

            # Forecast plot (show only selected model unless comparing both)
            fig1 = plt.figure(figsize=(12, 4))
            plt.plot(train, label="Train")
            plt.plot(test, label="Test")

            if model_type == "XGBoost":
                plt.plot(pred_xgb, "--", label="XGBoost")
            elif model_type == "ARIMA":
                plt.plot(pred_arima, ":", label="ARIMA")
            else:  # Compare Both
                plt.plot(pred_xgb, "--", label="XGBoost")
                plt.plot(pred_arima, ":", label="ARIMA")

            plt.title("Sales Forecast")
            plt.legend()
            forecast_image = make_image(fig1)


            # Top 10 Products by Sales (with forecast if possible)
            fig_top = plt.figure(figsize=(12, 4))
            if "Product Name" in data.columns:
                # Try to add forecasted sales by product for the forecast period
                data_cp = data.copy()
                data_cp[dcol] = pd.to_datetime(data_cp[dcol], errors="coerce")
                # Aggregate historical sales by product
                hist_top = data_cp.groupby("Product Name")[scol].sum()
                # Try to estimate forecasted sales by product for the forecast period
                # Only possible if Product Name is available for each forecasted period (not typical for time series)
                # So: Mark with a note
                # Note: Top Products is based on total historical sales, not forecasted
                top_products = hist_top.sort_values(ascending=False).head(10)
                ax = top_products.plot(kind="bar", color="skyblue")
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
            top_image = make_image(fig_top)

            # Sales by Segment or Region (use recent/forecast-relevant data)
            fig_seg = plt.figure(figsize=(12, 4))
            seg_sales = data.copy()
            seg_sales[dcol] = pd.to_datetime(seg_sales[dcol], errors="coerce")
            seg_sales = seg_sales[seg_sales[dcol] >= df.index[-months]]  # include only recent data
            seg_data = seg_sales.groupby(["Segment", "Region"])[scol].sum().unstack().fillna(0)
            seg_data.plot(kind="bar", stacked=False, ax=plt.gca())
            plt.title("Sales by Segment and Region")
            plt.ylabel("Sales")
            plt.xticks(rotation=0)
            segment_image = make_image(fig_seg)

            fig_profit_forecast = plt.figure(figsize=(12, 4))
            plt.plot(train_profit, label="Train Profit", color="green")
            plt.plot(test_profit, label="Test Profit", color="orange")

            if model_type == "XGBoost":
                plt.plot(pred_profit_xgb, "--", label="XGBoost Profit Forecast", color="red")
            elif model_type == "ARIMA":
                plt.plot(pred_profit_arima, ":", label="ARIMA Profit Forecast", color="purple")
            else:  # Compare Both
                plt.plot(pred_profit_xgb, "--", label="XGBoost Profit Forecast", color="red")
                plt.plot(pred_profit_arima, ":", label="ARIMA Profit Forecast", color="purple")

            plt.title("Forecasted Profit Over Time")
            plt.legend()
            profit_forecast_image = make_image(fig_profit_forecast)

            plot_views["Forecasted Sales"] = forecast_image
            plot_views["Forecasted Profit"] = profit_forecast_image
            plot_views["Top Products"] = top_image
            plot_views["Segment Sales"] = segment_image
            # Ensure the default tab displays the correct chart
            plot_forecast.content = plot_views["Forecasted Sales"]

            trend = "increase 📈" if pred_xgb.mean() > train[-months:].mean() else "decrease 📉"
            forecast_summary.value = f"🔎 Based on the forecast, sales are expected to {trend}."


            # Publish forecast data to dashboard and generate summary

            # --- Automation Tools Integration ---
            log_block = []

            run_ci_pipeline()
            log_block.append("✅ CI/CD pipeline executed.")


            # Removed simulated report email sending (no attachment)

            log_block.append("📊 Dashboard preview simulated (feature under development)")

            summary_text = f"XGBoost MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}"
            log_event(summary_text)
            log_block.append(f"🧠 Summary generated: {summary_text}")

            automation_output_text.value = "\n".join(log_block)
            # AI Insight Summary update
            summary_text = generate_ai_insight_summary(forecast_df)
            # Parse summary_text into lines and update dynamic AI summary texts, with bold markdown labels
            ai_lines = summary_text.splitlines()
            # Bold labeling for each field
            def extract_value(line, label):
                if line.startswith("-"):
                    # Remove leading "-" and whitespace
                    return line.lstrip("-").strip()
                return line
            if len(ai_lines) > 1:
                ai_summary_text_1.value = f"**Trend**: {extract_value(ai_lines[1], 'Trend')}"
            else:
                ai_summary_text_1.value = ""
            if len(ai_lines) > 2:
                ai_summary_text_2.value = f"**Highest Forecasted Value**: {extract_value(ai_lines[2], 'Highest forecasted value')}"
            else:
                ai_summary_text_2.value = ""
            if len(ai_lines) > 3:
                ai_summary_text_3.value = f"**Lowest Forecasted Value**: {extract_value(ai_lines[3], 'Lowest forecasted value')}"
            else:
                ai_summary_text_3.value = ""
            if len(ai_lines) > 4:
                ai_summary_text_4.value = f"**Forecast Span**: {extract_value(ai_lines[4], 'Forecast spans')}"
            else:
                ai_summary_text_4.value = ""
            if len(ai_lines) > 5:
                ai_summary_text_5.value = f"**Total Forecasted Revenue**: {extract_value(ai_lines[5], 'Total forecasted revenue')}"
            else:
                ai_summary_text_5.value = ""
            if len(ai_lines) > 6:
                ai_summary_text_6.value = f"**Average Sales per Period**: {extract_value(ai_lines[6], 'Average forecasted sales per period')}"
            else:
                ai_summary_text_6.value = ""
            if len(ai_lines) > 7:
                ai_summary_text_7.value = f"**Volatility (Std Dev)**: {extract_value(ai_lines[7], 'Standard deviation')}"
            else:
                ai_summary_text_7.value = ""
            # The following are for extra lines, if present in summary
            if len(ai_lines) > 8:
                ai_summary_text_8.value = extract_value(ai_lines[8], '')
            else:
                ai_summary_text_8.value = ""
            if len(ai_lines) > 9:
                ai_summary_text_9.value = extract_value(ai_lines[9], '')
            else:
                ai_summary_text_9.value = ""
            # forecast_summary.value += f"\n\n🔍 Summary:\n{summary_text}"
            download_csv_btn.visible = True
            download_excel_btn.visible = True
            show_snack("✅ Forecast complete!")
        except Exception as ex:
            file_info.value = f"❌ Forecast failed: {ex}"
            log_error(f"Exception during forecasting: {ex}")

        forecast_btn.disabled = False
        forecast_btn.text = "📈 Forecast"
        loading_indicator.visible = False
        page.update()

    # Load default dataset
    def load_default_dataset():
        nonlocal data
        try:
            path = os.path.join(os.path.dirname(__file__), "Dataset", "Dataset.csv")
            for enc in ["utf-8", "ISO-8859-1", "cp1252"]:
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            data, log = preprocess_data(df, column="Sales")
            log_output.value = "🧹 Data Cleaning Log:\n" + "\n".join(log)
            date_dropdown.options = [ft.dropdown.Option(c) for c in data.columns]
            sales_dropdown.options = [ft.dropdown.Option(c) for c in data.columns]
            # groupby_dropdown.options = [ft.dropdown.Option(c) for c in data.columns if data[c].dtype == "object"]
            date_dropdown.value = next((c for c in data.columns if "date" in c.lower()), data.columns[0])
            sales_dropdown.value = next((c for c in data.columns if "sale" in c.lower() or "amount" in c.lower()), data.columns[-1])
            freq_code, freq_label = detect_frequency(pd.to_datetime(data[date_dropdown.value], errors="coerce"))
            freq_info.value = f"📅 Detected frequency: {freq_label}"
            file_info.value = "✅ Default dataset loaded"
            plot_forecast.content = None  # clear any previous chart preview
            # Update smart insights after loading default dataset
            update_smart_insights()
            page.update()
            forecast(None)
        except Exception as e:
            file_info.value = f"❌ Failed to load default dataset: {e}"

    # upload_btn.on_click = lambda _: file_picker.pick_files(allow_multiple=False, allowed_extensions=["csv", "xlsx"])
    file_picker.on_result = load_file
    forecast_btn.on_click = forecast

    def download_csv(e):
        if forecast_csv:
            with open("forecast_output.csv", "w") as f:
                f.write(forecast_csv)
            os.system("open forecast_output.csv")

    def download_excel(e):
        if forecast_excel is not None:
            forecast_excel.to_excel("forecast_output.xlsx", index=False)
            os.system("open forecast_output.xlsx")

    download_csv_btn.on_click = download_csv
    download_excel_btn.on_click = download_excel

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
                    ft.Row([date_dropdown], alignment="center"),
                    ft.Row([
                        ft.Text("Sales Column", size=13, weight="medium"),
                        ft.IconButton(icon=ft.Icons.INFO_OUTLINE, tooltip="Select the numeric field that holds sales data.")
                    ]),
                    ft.Row([sales_dropdown], alignment="center"),
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
                    ft.Row([model_dropdown], alignment="center"),
                    ft.Row([
                        ft.Text("Periods to Forecast", size=13, weight="medium"),
                        ft.IconButton(icon=ft.Icons.INFO_OUTLINE, tooltip="How many future time periods to forecast?")
                    ]),
                    ft.Row([months_slider], alignment="center")
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
                    ft.Row([forecast_btn], alignment="center"),
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
                    file_info,
                    ft.Text("⏳ Frequency:", size=13, weight="medium"),
                    freq_info,
                    ft.Text("📌 Cleaning Log", weight="bold"),
                    log_output,
                ]),
                bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.BLUE_GREY),
                padding=10,
                border_radius=8
            )
        ], scroll=ft.ScrollMode.AUTO)
    )

    background_container = ft.Container(
        expand=True,
        gradient=get_gradient(),
        content=ft.Column([
            ft.Container(
                ft.Row([
                    ft.Text("Sales Forecasting Dashboard", size=22, weight="bold"),
                    ft.Row([
                        ft.Switch(label="🌙 Dark Mode", value=False, on_change=toggle_dark)
                    ])
                ], alignment="spaceBetween"),
                padding=15,
                bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.BLUE_GREY)
            ),
            ft.Text("📦 Welcome to the Store Sales Forecasting Application!", size=14, italic=True),
            # Removed smart insights container from here
            ft.Row([
                sidebar,
                ft.Container(expand=True, content=ft.Column([
                    ft.Row([loading_indicator], alignment="center"),
                    # Project info/status/tip row
                    ft.Container(
                        content=ft.Row([
                            ft.Text("🧠 Project: Store Sales Forecasting", size=14, weight="bold"),
                            ft.Text("🔄 Status: Ready", size=13, italic=True),
                            ft.Text("💡 Tip: Use 'Compare Both' to validate model performance!", size=13, italic=True),
                        ], alignment="spaceAround"),
                        padding=10
                    ),
                    result_metrics,
                    forecast_summary,
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
                                    highest_month_text,
                                    lowest_quarter_text,
                                ], spacing=10, expand=True, alignment="center"),
                                ft.Column([
                                    trend_text,
                                    top_region_text,
                                ], spacing=10, expand=True, alignment="center"),
                                ft.Column([
                                    most_profitable_product_text,
                                    least_selling_product_text,
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
                            ft.ResponsiveRow([
                                ft.Column([
                                    ft.Text("Trend:", weight="bold"),
                                    ai_summary_text_1
                                ], col={"sm": 12, "md": 6}),
                                ft.Column([
                                    ft.Text("Highest Forecasted Value:", weight="bold"),
                                    ai_summary_text_2
                                ], col={"sm": 12, "md": 6}),
                                ft.Column([
                                    ft.Text("Lowest Forecasted Value:", weight="bold"),
                                    ai_summary_text_3
                                ], col={"sm": 12, "md": 6}),
                                ft.Column([
                                    ft.Text("Forecast Span:", weight="bold"),
                                    ai_summary_text_4
                                ], col={"sm": 12, "md": 6}),
                                ft.Column([
                                    ft.Text("Total Forecasted Revenue:", weight="bold"),
                                    ai_summary_text_5
                                ], col={"sm": 12, "md": 6}),
                                ft.Column([
                                    ft.Text("Average Sales per Period:", weight="bold"),
                                    ai_summary_text_6
                                ], col={"sm": 12, "md": 6}),
                                ft.Column([
                                    ft.Text("Volatility (Std Dev):", weight="bold"),
                                    ai_summary_text_7
                                ], col={"sm": 12, "md": 6}),
                            ], spacing=10)
                        ]),
                        padding=20,
                        bgcolor=ft.Colors.with_opacity(0.06, ft.Colors.GREEN),
                        border_radius=12,
                        margin=ft.margin.only(top=10, bottom=10)
                    ),
                    ft.Container(
                        content=ft.Column([
                            
                            email_history_text
                        ]),
                        bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.BLUE),
                        padding=15,
                        border_radius=10,
                        margin=ft.margin.only(top=5, bottom=10)
                    ),
                    ft.Row([download_csv_btn, download_excel_btn], alignment="center"),
                    ft.Row([email_input, send_email_btn], alignment="center"),
                    # Insert tab buttons in a row (fullscreen button removed)
                    ft.Row([forecast_tab_btns], alignment="center"),
                    # Make plot_forecast scrollable and expandable for multi-plot comparison
                    ft.Row([
                        ft.Container(content=ft.Column([plot_forecast], scroll=ft.ScrollMode.AUTO), expand=True)
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

    load_default_dataset()
    page.add(background_container)

# --- AI Insight Summary Generator ---
def generate_ai_insight_summary(df):
    if df.empty:
        return "No insights available due to empty forecast data."
    try:
        sales_forecast = df["XGBoost Forecast"] if "XGBoost Forecast" in df.columns else df.iloc[:, 1]
        trend = "rising 📈" if sales_forecast.iloc[-1] > sales_forecast.iloc[0] else "declining 📉"
        max_forecast = sales_forecast.max()
        min_forecast = sales_forecast.min()
        return (
            f"- Predicted sales show a {trend} trend.\n"
            f"- Highest forecasted value: {max_forecast:.2f}\n"
            f"- Lowest forecasted value: {min_forecast:.2f}\n"
            f"- Forecast spans {len(sales_forecast)} periods.\n"
            f"- Total forecasted revenue: {sales_forecast.sum():.2f}\n"
            f"- Average forecasted sales per period: {sales_forecast.mean():.2f}\n"
            f"- Standard deviation (volatility): {sales_forecast.std():.2f}"
        )
    except Exception as e:
        return f"⚠️ Failed to generate insight summary: {e}"

if __name__ == "__main__":
        app(target=main , view=ft.WEB_BROWSER)