import logging
import os

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/app_forecast_log.txt",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add console logging as well
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

def logInfo(message):
    logging.info(message)

def logWarning(message):
    logging.warning(message)

def logError(message):
    logging.error(message)

def logDebug(message):
    logging.debug(message)

def logData(tag, data):
    logging.info(f"{tag} - {repr(data)}")

def logForecastReady():
    logging.info("✅ Forecast output is ready to be downloaded via the GUI.")
