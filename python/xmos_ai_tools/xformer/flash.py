BYTES_FOR_MAGIC_PATTERN = 32
BYTES_FOR_HEADER = 4
BYTES_PER_ENGINE_BLOCK = 16
VERSION_MAJOR = 1
VERSION_MINOR = 2


class FlashBuilder:
    class Header:
        """
        Class that stores a header for a flash file system
        The header comprises the addresses of the model, parameters, and operators
        relative to the start address
        """

        def __init__(self, model_bytes, params_bytes, ops_bytes, xip_bytes, start):
            self.model_start = start
            self.parameters_start = self.model_start + model_bytes + 4  # len
            self.operators_start = self.parameters_start + params_bytes  # no len
            self.xip_start = self.operators_start + ops_bytes  # no len
            new_start = self.xip_start + xip_bytes  # no len
            self.length = new_start - start

    def __init__(self, engines=1):
        self.engines = engines
        self.models = [bytes([])] * engines
        self.params = [bytes([])] * engines
        self.ops = [bytes([])] * engines
        self.xips = [bytes([])] * engines

    @staticmethod
    def read_whole_binary_file(filename):
        """
        Reads a whole binary file in and returns bytes(). If the file to be read is called '-'
        then an empty bytes is returned.
        """
        if filename == "-":
            return bytes([])
        try:
            with open(filename, "rb") as input_fd:
                contents = input_fd.read()
            return contents
        except:
            print('File "%s" is not a readable file' % (filename))
            return None

    @staticmethod
    def create_params_image(params=None, filename=None):
        if params is None:
            params = FlashBuilder.read_whole_binary_file(filename)
        return params

    @staticmethod
    def create_model_image(model=None, filename=None):
        if model is None:
            model = FlashBuilder.read_whole_binary_file(filename)
        return model

    @staticmethod
    def create_params_file(params_filename, params=None, input_filename=None):
        image = FlashBuilder.create_params_image(params=params, filename=input_filename)
        with open(params_filename, "wb") as output_fd:
            output_fd.write(image)

    @staticmethod
    def tobytes(integr):
        """Converts an int to a LSB first quad of bytes"""
        data = []
        for i in range(4):
            data.append((integr >> (8 * i)) & 0xFF)
        return bytes(data)

    @staticmethod
    def swap_nibbles(x):
        return ( (x & 0x0F)<<4 | (x & 0xF0)>>4 )

    def add_params(self, engine, params=None, filename=None):
        image = FlashBuilder.create_params_image(params, filename)
        self.params[engine] = image

    def add_model(self, engine, model=None, filename=None):
        image = FlashBuilder.create_model_image(model, filename)
        self.models[engine] = image

    def flash_image(self):
        """
        Builds a flash image out of a collection of models and parameter blobs.
        This function returns a bytes comprising the header, models, parameters, etc.
        The whole thing should be written as is to flash
        """
        headers = [None] * self.engines
        start = BYTES_FOR_MAGIC_PATTERN + BYTES_FOR_HEADER + BYTES_PER_ENGINE_BLOCK * self.engines
        for i in range(self.engines):
            headers[i] = FlashBuilder.Header(
                len(self.models[i]),
                len(self.params[i]),
                len(self.ops[i]),
                len(self.xips[i]),
                start,
            )
            start += headers[i].length

        # We add the magic fast flash pattern of 32 bytes at the very beginning
        # After that comes the version
        output = bytes(
            [0xff, 0x00, 0x0f, 0x0f,
            0x0f, 0x0f, 0x0f, 0x0f,
            0xff, 0x00, 0xff, 0x00,
            0xff, 0x00, 0xff, 0x00,
            0x31, 0xf7, 0xce, 0x08,
            0x31, 0xf7, 0xce, 0x08,
            0x9c, 0x63, 0x9c, 0x63,
            0x9c, 0x63, 0x9c, 0x63])

        output += bytes(
            [VERSION_MAJOR, VERSION_MINOR, 0xFF ^ VERSION_MAJOR, 0xFF ^ VERSION_MINOR]
        )

        for i in range(self.engines):
            output += FlashBuilder.tobytes(
                headers[i].model_start
            )  # encode start of model
            output += FlashBuilder.tobytes(
                headers[i].parameters_start
            )  # encode start of params
            output += FlashBuilder.tobytes(
                headers[i].operators_start
            )  # encode start of ops
            output += FlashBuilder.tobytes(headers[i].xip_start)  # encode start of xip

        for i in range(self.engines):
            output += FlashBuilder.tobytes(len(self.models[i]))  # encode len of model
            output += self.models[i]  # add model image
            output += self.params[i]  # add params image
            output += self.ops[i]  # add operators image
            output += self.xips[i]  # add exec in place image
        return output

    def flash_file(self, filename):
        """
        Builds a file for the host system that comprises a single parameter blob.
        """
        output = self.flash_image()
        # swap nibbles around
        swapped_output = bytes([FlashBuilder.swap_nibbles(byte) for byte in output])
        with open(filename, "wb") as output_fd:
            output_fd.write(swapped_output)


def generate_flash(model_file, params_file, output_file):
    fb = FlashBuilder()
    fb.add_model(0, filename=model_file)
    fb.add_params(0, filename=params_file)
    fb.flash_file(output_file)


