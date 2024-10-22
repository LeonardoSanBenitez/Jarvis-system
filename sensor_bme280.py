#--------------------------------------
#  Read data from a digital pressure sensor.
#
#  Official datasheet available from :
#  https://www.bosch-sensortec.com/bst/products/all_products/bme280
#
# @Author     : Matt Hawkins
# @Adapted by : Leonardo Benitez
# @Date       : 21/01/2018
#
# https://www.raspberrypi-spy.co.uk/
#
#--------------------------------------

from ctypes import c_ubyte
from ctypes import c_byte
from ctypes import c_short

import smbus
import time

class Bme280 ():
  _bus = None
  _device = None

  def __init__(self):
    self._device = 0x76  # Default device I2C address
    # Rev 2 Pi, Pi 2 & Pi 3 uses bus 1 (Rev 1 Pi uses bus 0)
    self._bus = smbus.SMBus(1)

  def getShort(self, data, index):
    # return two bytes from data as a signed 16-bit value
    return c_short((data[index+1] << 8) + data[index]).value

  def getUShort(self, data, index):
    # return two bytes from data as an unsigned 16-bit value
    return (data[index+1] << 8) + data[index]

  def getChar(self, data, index):
    # return one byte from data as a signed char
    result = data[index]
    if result > 127:
      result -= 256
    return result

  def getUChar(self, data, index):
    # return one byte from data as an unsigned char
    result = data[index] & 0xFF
    return result

  def readBME280ID(self):
    # Chip ID Register Address
    REG_ID = 0xD0
    (chip_id, chip_version) = self._bus.read_i2c_block_data(self._device, REG_ID, 2)
    return (chip_id, chip_version)

  def readBME280All(self):
    # Register Addresses
    REG_DATA = 0xF7
    REG_CONTROL = 0xF4
    REG_CONFIG = 0xF5

    REG_CONTROL_HUM = 0xF2
    REG_HUM_MSB = 0xFD
    REG_HUM_LSB = 0xFE

    # Oversample setting - page 27
    OVERSAMPLE_TEMP = 2
    OVERSAMPLE_PRES = 2
    MODE = 1

    # Oversample setting for humidity register - page 26
    OVERSAMPLE_HUM = 2
    self._bus.write_byte_data(self._device, REG_CONTROL_HUM, OVERSAMPLE_HUM)

    control = OVERSAMPLE_TEMP << 5 | OVERSAMPLE_PRES << 2 | MODE
    self._bus.write_byte_data(self._device, REG_CONTROL, control)

    # Read blocks of calibration data from EEPROM
    # See Page 22 data sheet
    cal1 = self._bus.read_i2c_block_data(self._device, 0x88, 24)
    cal2 = self._bus.read_i2c_block_data(self._device, 0xA1, 1)
    cal3 = self._bus.read_i2c_block_data(self._device, 0xE1, 7)

    # Convert byte data to word values
    dig_T1 = self.getUShort(cal1, 0)
    dig_T2 = self.getShort(cal1, 2)
    dig_T3 = self.getShort(cal1, 4)

    dig_P1 = self.getUShort(cal1, 6)
    dig_P2 = self.getShort(cal1, 8)
    dig_P3 = self.getShort(cal1, 10)
    dig_P4 = self.getShort(cal1, 12)
    dig_P5 = self.getShort(cal1, 14)
    dig_P6 = self.getShort(cal1, 16)
    dig_P7 = self.getShort(cal1, 18)
    dig_P8 = self.getShort(cal1, 20)
    dig_P9 = self.getShort(cal1, 22)

    dig_H1 = self.getUChar(cal2, 0)
    dig_H2 = self.getShort(cal3, 0)
    dig_H3 = self.getUChar(cal3, 2)

    dig_H4 = self.getChar(cal3, 3)
    dig_H4 = (dig_H4 << 24) >> 20
    dig_H4 = dig_H4 | (self.getChar(cal3, 4) & 0x0F)

    dig_H5 = self.getChar(cal3, 5)
    dig_H5 = (dig_H5 << 24) >> 20
    dig_H5 = dig_H5 | (self.getUChar(cal3, 4) >> 4 & 0x0F)

    dig_H6 = self.getChar(cal3, 6)

    # Wait in ms (Datasheet Appendix B: Measurement time and current calculation)
    wait_time = 1.25 + (2.3 * OVERSAMPLE_TEMP) + ((2.3 *
                                                   OVERSAMPLE_PRES) + 0.575) + ((2.3 * OVERSAMPLE_HUM)+0.575)
    time.sleep(wait_time/1000)  # Wait the required time

    # Read temperature/pressure/humidity
    data = self._bus.read_i2c_block_data(self._device, REG_DATA, 8)
    pres_raw = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4)
    temp_raw = (data[3] << 12) | (data[4] << 4) | (data[5] >> 4)
    hum_raw = (data[6] << 8) | data[7]

    #Refine temperature
    var1 = ((((temp_raw >> 3)-(dig_T1 << 1)))*(dig_T2)) >> 11
    var2 = (((((temp_raw >> 4) - (dig_T1)) *
              ((temp_raw >> 4) - (dig_T1))) >> 12) * (dig_T3)) >> 14
    t_fine = var1+var2
    temperature = float(((t_fine * 5) + 128) >> 8)

    # Refine pressure and adjust for temperature
    var1 = t_fine / 2.0 - 64000.0
    var2 = var1 * var1 * dig_P6 / 32768.0
    var2 = var2 + var1 * dig_P5 * 2.0
    var2 = var2 / 4.0 + dig_P4 * 65536.0
    var1 = (dig_P3 * var1 * var1 / 524288.0 + dig_P2 * var1) / 524288.0
    var1 = (1.0 + var1 / 32768.0) * dig_P1
    if var1 == 0:
      pressure = 0
    else:
      pressure = 1048576.0 - pres_raw
      pressure = ((pressure - var2 / 4096.0) * 6250.0) / var1
      var1 = dig_P9 * pressure * pressure / 2147483648.0
      var2 = pressure * dig_P8 / 32768.0
      pressure = pressure + (var1 + var2 + dig_P7) / 16.0

    # Refine humidity
    humidity = t_fine - 76800.0
    humidity = (hum_raw - (dig_H4 * 64.0 + dig_H5 / 16384.0 * humidity)) * (dig_H2 / 65536.0 *
                                                                            (1.0 + dig_H6 / 67108864.0 * humidity * (1.0 + dig_H3 / 67108864.0 * humidity)))
    humidity = humidity * (1.0 - dig_H1 * humidity / 524288.0)
    if humidity > 100:
      humidity = 100
    elif humidity < 0:
      humidity = 0

    dict = {
        "temperature": temperature/100.0,
        "pressure": pressure/100.0,
        "humidity": humidity
    }
    return dict
