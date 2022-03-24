	// >>>>>>>>>>> 第一部分
    {
		.name		= "Depth data 16-bit (D16)",
		.guid		= UVC_GUID_FORMAT_D16,
		.fcc		= V4L2_PIX_FMT_Z16,
	},
	{
		.name		= "Packed raw data 10-bit",
		.guid		= UVC_GUID_FORMAT_W10,
		.fcc		= V4L2_PIX_FMT_W10,
	},
	{
		.name		= "Confidence data (C   )",
		.guid		= UVC_GUID_FORMAT_CONFIDENCE_MAP,
		.fcc		= V4L2_PIX_FMT_CONFIDENCE_MAP,
	},
	/* FishEye 8-bit monochrome */
	{
		.name		= "Raw data 8-bit (RAW8)",
		.guid		= UVC_GUID_FORMAT_RAW8,
		.fcc		= V4L2_PIX_FMT_GREY,
	},
	/* Legacy formats for backward-compatibility*/
	{
		.name		= "Raw data 16-bit (RW16)",
		.guid		= UVC_GUID_FORMAT_RW16,
		.fcc		= V4L2_PIX_FMT_RW16,
	},
	{
		.name		= "16-bit Bayer BGBG/GRGR",
		.guid		= UVC_GUID_FORMAT_BAYER16,
		.fcc		= V4L2_PIX_FMT_SBGGR16,
	},
	{
		.name		= "Z16 Huffman Compression",
		.guid		= UVC_GUID_FORMAT_Z16H,
		.fcc		= V4L2_PIX_FMT_Z16H,
	},
	{
		.name		= "Frame Grabber (FG  )",
		.guid		= UVC_GUID_FORMAT_FG,
		.fcc		= V4L2_PIX_FMT_FG,
	},
	{
		.name		= "SR300 Depth/Confidence (INZC)",
		.guid		= UVC_GUID_FORMAT_INZC,
		.fcc		= V4L2_PIX_FMT_INZC,
	},
	{
		.name		= "Relative IR (PAIR)",
		.guid		= UVC_GUID_FORMAT_PAIR,
		.fcc		= V4L2_PIX_FMT_PAIR,
	},
    
    
    // >>>>>>>>>>> 第二部分
    /* Intel RealSense D4M */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor		= 0x8086,
	  .idProduct		= 0x0b03,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel SR306 depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0aa3,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel SR300 depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0aa5,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	/* Intel D400/PSR depth camera*/
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0ad1,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D410/ASR depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0ad2,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D415/ASRC depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0ad3,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D430/AWG depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0ad4,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	/* Intel D450/AWGT depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0ad5,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	/* USB2 Descriptor, Depth Sensor */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0ad6,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D400 IMU Module */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0af2,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	/* Intel D420/PWG depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0af6,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	/* Intel D420_MM/PWGT depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0afe,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D410_MM/ASRT depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0aff,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D400_MM/PSRT depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b00,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D430_MM/AWGCT depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b01,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D460/DS5U depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b03,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D435/AWGC depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b07,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D405 S depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b0c,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel L500 depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b0d,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D435i depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b3a,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel L515 Pre-PRQ */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b3d,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel SR305 Depth Camera*/
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b48,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D416 Depth Camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b49,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D430i depth camera */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b4b,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D465 */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b4d,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D405 */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b5b,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel D455 */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b5c,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel L515 */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor			= 0x8086,
	  .idProduct		= 0x0b64,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },
	  /* Intel L535 */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor		= 0x8086,
	  .idProduct		= 0x0b68,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_INFO_META(V4L2_META_FMT_D4XX) },