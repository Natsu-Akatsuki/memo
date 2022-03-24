/* Capture samples in local storage */
static int accel_3d_capture_sample(struct hid_sensor_hub_device *hsdev,
				unsigned usage_id,
				size_t raw_len, char *raw_data,
				void *priv)
{
	struct iio_dev *indio_dev = platform_get_drvdata(priv);
	struct accel_3d_state *accel_state = iio_priv(indio_dev);
	int offset;
	int ret = -EINVAL;

	switch (usage_id) {
	case HID_USAGE_SENSOR_ACCEL_X_AXIS:
	case HID_USAGE_SENSOR_ACCEL_Y_AXIS:
	case HID_USAGE_SENSOR_ACCEL_Z_AXIS:
		offset = usage_id - HID_USAGE_SENSOR_ACCEL_X_AXIS;
		accel_state->scan.accel_val[CHANNEL_SCAN_INDEX_X + offset] =
						*(u32 *)raw_data;
		ret = 0;
	break;
	case HID_USAGE_SENSOR_TIME_TIMESTAMP:
		accel_state->timestamp =
			hid_sensor_convert_timestamp(
					&accel_state->common_attributes,
					*(int64_t *)raw_data);
		ret = 0;
	break;
	default:
		break;
	}

	return ret;
}
