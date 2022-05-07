#include "serial.hpp"

uint8_t serial::crc8x_cal(const uint8_t *mem, size_t len) {
  uint8_t crc = 0x00;
  const uint8_t *data = mem;
  if (data == NULL)
      return 0xff;
  crc &= 0xff;
  while (len--)
      crc = crc8x_table[crc ^ *data++];
  return crc;
}