from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService

from PIL import Image
from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer, ImageLayer


class RecorderExample(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService(transcription_model='base'))
        # self.set_speech_service(RecorderService())

        with self.voiceover(text="This is a 101 on Quantum Error Correcting Autoencoders.") as tracker:
            title = Text("Quantum Error Correcting Autoencoders")
            self.play(Write(title), run_time=tracker.duration)
            self.play(FadeOut(title))

        with self.voiceover(text="First let's review the concepts of a classical autoencoder to understand the motivation for turning it quantum. <bookmark mark='A'/> The autoencoder is a subclass of deep neural networks in which the network is trained to output its input. <bookmark mark='B'/> Autoencoders tend to have a small hidden layer, called the latent space, which allows for a compact representation of the input. <bookmark mark='C'/> This means you can seperate the encoder and decoder, leaving the latent space as a compressed representation of the original content. <bookmark mark='D'/> Further, autoencoders can be used for simultaneous denoising and compression, by training the model to return good images even with noisy inputs. <bookmark mark='E'/>") as tracker:
            numpy_image = np.asarray(Image.open("assets/digit.jpeg"))
            nn = NeuralNetwork(
                        [
                            # ImageLayer(numpy_image, height=1.4),
                            FeedForwardLayer(5),
                            FeedForwardLayer(3),
                            FeedForwardLayer(3),
                            FeedForwardLayer(5),
                            # ImageLayer(numpy_image, height=1.4),
                        ]
                    )
            nn.scale(2)
            self.play(FadeIn(nn))

            self.wait_until_bookmark("A")
            numpy_image = np.asarray(Image.open("assets/digit.jpeg"))
            nn_with_image = NeuralNetwork(
                        [
                            ImageLayer(numpy_image, height=1.4),
                            FeedForwardLayer(10),
                            FeedForwardLayer(5),
                            FeedForwardLayer(3),
                            FeedForwardLayer(5),
                            FeedForwardLayer(10),
                            ImageLayer(numpy_image, height=1.4),
                        ]
                    )
            nn_with_image.scale(2)
            self.play(FadeOut(nn))
            self.play(FadeIn(nn_with_image))

            self.wait_until_bookmark("B")
            arrow = Arrow(start=DOWN*2, end=DOWN*1, stroke_width=16, color=WHITE)
            self.play(Create(arrow))

            self.wait_until_bookmark("C")
            self.play(FadeOut(nn_with_image))
            self.play(FadeOut(arrow))
            encoder = NeuralNetwork(
                        [
                            ImageLayer(numpy_image, height=1.4),
                            FeedForwardLayer(10),
                            FeedForwardLayer(5),
                            FeedForwardLayer(3),
                        ]
            )
            encoder.scale(2)
            encoder.shift(LEFT*4)
            decoder = NeuralNetwork(
                        [
                            FeedForwardLayer(3),
                            FeedForwardLayer(5),
                            FeedForwardLayer(10),
                            ImageLayer(numpy_image, height=1.4),
                        ]
            )
            decoder.scale(2)
            decoder.shift(RIGHT*4)
            self.play(FadeIn(encoder))
            self.play(FadeIn(decoder))

            self.wait_until_bookmark("D")
            self.play(FadeOut(encoder), FadeOut(decoder))
            noisy_image = numpy_image + np.random.normal(size=numpy_image.shape)
            nn_with_noisy_image = NeuralNetwork(
                        [
                            ImageLayer(noisy_image, height=1.4),
                            FeedForwardLayer(10),
                            FeedForwardLayer(5),
                            FeedForwardLayer(3),
                            FeedForwardLayer(5),
                            FeedForwardLayer(10),
                            ImageLayer(numpy_image, height=1.4),
                        ]
            )
            nn_with_noisy_image.scale(2)
            self.play(FadeIn(nn_with_noisy_image))

            self.wait_until_bookmark("E")
            self.play(FadeOut(nn_with_noisy_image))


        with self.voiceover(text="In quantum computing, two major challanges are compression of data are efficently storing data in fewer qbits, and error correcting. Autoencoders can solve both at once, and therefore are a very natural archecture to adapt. <bookmark mark='A'/> I will be brief with many explenations in the adaptation of classical to quantum autoencoder, so I will try to leave references down on the bottom right for fruther exploration. <bookmark mark='B'/> There are three key steps for training a usuable autoencoder.<bookmark mark='C'/> The forward pass. <bookmark mark='D'/> The backpropagation. <bookmark mark='D'/> And the optimizer. <bookmark mark='E'/> We already run into an issue trying to make a quantum forward pass: Neural networks rely on data cloning for ") as tracker:
            image = ImageMobject("assets/problems.png")
            self.play(FadeIn(image))

            self.wait_until_bookmark("A")
            self.play(FadeOut(image))
            text = Text("Doersch, C. (2016). Tutorial on variational autoencoders. arXiv preprint arXiv:1606.05908.", font_size=16)
            text.to_corner(DR)
            self.play(Write(text))

            self.wait_until_bookmark("B")
            numpy_image = np.asarray(Image.open("assets/digit.jpeg"))
            nn = NeuralNetwork(
                        [
                            ImageLayer(numpy_image, height=1.4),
                            FeedForwardLayer(10),
                            FeedForwardLayer(5),
                            FeedForwardLayer(3),
                            FeedForwardLayer(5),
                            FeedForwardLayer(10),
                        ]
            )
            nn.scale(2)
            self.play(FadeIn(nn))

            self.wait_until_bookmark("C")
            self.play(FadeOut(nn))
            nn = NeuralNetwork(
                        [
                            ImageLayer(numpy_image, height=1.4),
                            FeedForwardLayer(10),
                            FeedForwardLayer(5),
                            FeedForwardLayer(3),
                            FeedForwardLayer(5),
                            FeedForwardLayer(10),
                            ImageLayer(np.random.normal(size=numpy_image.shape), height=1.4),
                        ]
            )
            nn.scale(2)
            self.play(FadeIn(nn))

            self.wait_until_bookmark("C")
            text = MathTex(r"\frac{\partial w_i^j}{\partial C}")
            text.shift(UP*2)
            self.play(Write(text))

            self.wait_until_bookmark("D")
            self.play(FadeOut(nn))
            nn = NeuralNetwork(
                        [
                            ImageLayer(numpy_image, height=1.4),
                            FeedForwardLayer(10),
                            FeedForwardLayer(5),
                            FeedForwardLayer(3),
                            FeedForwardLayer(5),
                            FeedForwardLayer(10),
                            ImageLayer(numpy_image, height=1.4),
                        ]
            )
            nn.scale(2)
            self.play(FadeIn(nn))
