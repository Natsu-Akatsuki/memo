// >>>>>>>>>>> 第一部分
struct gyro_3d_state {
	struct hid_sensor_hub_callbacks callbacks;
	struct hid_sensor_common common_attributes;
	struct hid_sensor_hub_attribute_info gyro[GYRO_3D_CHANNEL_MAX];
	/* Reserve for 3 channels + padding + timestamp */
	u32 gyro_val[GYRO_3D_CHANNEL_MAX + 3];
	int scale_pre_decml;
	int scale_post_decml;
	int scale_precision;
	int value_offset;
	int64_t timestamp;
};

// >>>>>>>>>>> 第二部分
{
    .type = IIO_ANGL_VEL,
    .modified = 1,
    .channel2 = IIO_MOD_Z,
    .info_mask_separate = BIT(IIO_CHAN_INFO_RAW),
    .info_mask_shared_by_type = BIT(IIO_CHAN_INFO_OFFSET) |
    BIT(IIO_CHAN_INFO_SCALE) |
    BIT(IIO_CHAN_INFO_SAMP_FREQ) |
    BIT(IIO_CHAN_INFO_HYSTERESIS),
    .scan_index = CHANNEL_SCAN_INDEX_Z,
},
IIO_CHAN_SOFT_TIMESTAMP(3)

// >>>>>>>>>>> 第三部分
/* Function to push data to buffer */
static void hid_sensor_push_data(struct iio_dev *indio_dev, void *data,
	int len, int64_t timestamp)
{
	dev_dbg(&indio_dev->dev, "hid_sensor_push_data\n");
	iio_push_to_buffers_with_timestamp(indio_dev, data, timestamp);
}

/* Callback handler to send event after all samples are received and captured */
static int gyro_3d_proc_event(struct hid_sensor_hub_device *hsdev,
				unsigned usage_id,
				void *priv)
{
	struct iio_dev *indio_dev = platform_get_drvdata(priv);
	struct gyro_3d_state *gyro_state = iio_priv(indio_dev);

	dev_dbg(&indio_dev->dev, "gyro_3d_proc_event\n");
	if (atomic_read(&gyro_state->common_attributes.data_ready)) {
		if (!gyro_state->timestamp)
			gyro_state->timestamp = iio_get_time_ns(indio_dev);

		hid_sensor_push_data(indio_dev,
				gyro_state->gyro_val,
				sizeof(gyro_state->gyro_val),
				gyro_state->timestamp);
		gyro_state->timestamp = 0;
	}
	return 0;
}

// >>>>>>>>>>> 第四部分
/* Capture samples in local storage */
static int gyro_3d_capture_sample(struct hid_sensor_hub_device *hsdev,
				unsigned usage_id,
				size_t raw_len, char *raw_data,
				void *priv)
{
	struct iio_dev *indio_dev = platform_get_drvdata(priv);
	struct gyro_3d_state *gyro_state = iio_priv(indio_dev);
	int offset;
	int ret = -EINVAL;

	switch (usage_id) {
	case HID_USAGE_SENSOR_ANGL_VELOCITY_X_AXIS:
	case HID_USAGE_SENSOR_ANGL_VELOCITY_Y_AXIS:
	case HID_USAGE_SENSOR_ANGL_VELOCITY_Z_AXIS:
		offset = usage_id - HID_USAGE_SENSOR_ANGL_VELOCITY_X_AXIS;
		gyro_state->gyro_val[CHANNEL_SCAN_INDEX_X + offset] =
						*(u32 *)raw_data;
		ret = 0;
	break;
	case HID_USAGE_SENSOR_TIME_TIMESTAMP:
		gyro_state->timestamp =
			hid_sensor_convert_timestamp(
					&gyro_state->common_attributes,
					*(int64_t *)raw_data);
		ret = 0;
	break;
	default:
		break;
	}

	return ret;
}