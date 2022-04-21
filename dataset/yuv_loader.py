# third-party packages
import numpy as np

def load_yuv_seq(seq_path, h, w, tot_frm, bit, start_frm=0):
	"""
	loader for yuv 4:2:0 data
	"""

	def load_8bit(seq_path, h, w, tot_frm, start_frm):
		# set up params
		blk_size = h * w * 3 // 2
		hh, ww = h // 2, w // 2

		# init
		y_seq = np.zeros((tot_frm, h, w), dtype=np.uint8)
		u_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)
		v_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)

		# read data
		with open(seq_path, 'rb') as fp:
			fp.seek(int(blk_size * start_frm), 0)

			for i in range(tot_frm):
				# print('{:3d} | {:3d}'.format(i + 1, tot_frm), end='\r')

				y_frm = np.fromfile(fp, dtype=np.uint8, count=h*w).reshape(h, w)
				u_frm = np.fromfile(fp, dtype=np.uint8, count=hh*ww).reshape(hh, ww)
				v_frm = np.fromfile(fp, dtype=np.uint8, count=hh*ww).reshape(hh, ww)

				y_seq[i, ...] = y_frm
				u_seq[i, ...] = u_frm
				v_seq[i, ...] = v_frm
			# print()

		return y_seq, u_seq, v_seq
		
	if bit == 8:
		import_fn = load_8bit
	else:
		raise NotImplementedError('Unknown bit depth: {}'.format(bit))

	y_seq, u_seq, v_seq = import_fn(seq_path, h, w, tot_frm, start_frm)

	return y_seq, u_seq, v_seq