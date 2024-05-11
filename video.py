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
                            FeedForwardLayer(5),
                            FeedForwardLayer(3),
                            FeedForwardLayer(3),
                            FeedForwardLayer(5),
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


        with self.voiceover(text="In quantum computing, two major challenges are compression of data are efficiently storing data in fewer qubits, and error correcting. Autoencoders can solve both at once, and therefore are a very natural architecture to adapt. <bookmark mark='A'/> I will be brief with many explanations in the adaptation of classical to quantum autoencoder, so I will try to leave references down on the bottom right for further exploration. <bookmark mark='B'/> There are three key steps for training a usable autoencoder.<bookmark mark='C'/> The forward pass. <bookmark mark='D'/> The backpropagation. <bookmark mark='D'/> And the optimizer. We already run into an issue trying to make a quantum forward pass: Neural networks rely on data cloning for the weights part of the forward pass. We want to recreate the style of linear transformation that weights and biases provide, which can be achieved using a parameterized n-qubit unitary gate. <bookmark mark='E'/> In this example, we will focus on the computational basis, so we can reduce the number of parameters by parameterizing with rotation gates. <bookmark mark='F'/> To decrease the size of our latent space, we can chop off qbits. Now we have a working forward pass, we need our backwards pass (finding the derivative of the weights with respect to a cost function). <bookmark mark='G'/> Our cost function will be the average fidelity between the input and output across the training data. <bookmark mark='H'/> However, we can leverage the fact that RealAmplitudes is unitary to half the number of nessesary parameters: we can use the inverse of the encoder block as our decoder block. In this senario, if the encoder block makes the bottom two qbits equal to the |0>, then the decoder block will be able to perfectly reconstruct the initial state. <bookmark mark='I'/> Therefore, we can greatly simplify the cost function by using the SWAP test, which will measure a lower value if the bottom qbits are more similar. It can also be run in hardware, rather than simulation, since it works on measurment rather than knowing the state. So the cost function is just getting the measurment block to measure 0 on the training data as often as possible. However, these quantum cost functions do not provide a way for backpropigation, so we don't know the derivative of the parameters with respect to the cost function. <bookmark mark='J'/> Instead, we can use a derivative free optimizer, like COBYLA. <bookmark mark='K'/>") as tracker:
            image = ImageMobject("assets/problems.png")
            self.play(FadeIn(image))

            self.wait_until_bookmark("A")
            self.play(FadeOut(image))
            cite = Text("Doersch, C. (2016). Tutorial on variational autoencoders. arXiv preprint arXiv:1606.05908.", font_size=16)
            cite.to_corner(DR)
            self.play(Write(cite))

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
            dc = MathTex(r"\frac{\partial w_i^j}{\partial C}")
            dc.shift(UP*2)
            self.play(Write(dc))

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

            self.wait_until_bookmark("E")
            self.play(FadeOut(nn))
            self.play(FadeOut(dc))
            self.play(Transform(cite, Text("Locher, D. F., Cardarelli, L., & Müller, M. (2023). Quantum error correction with quantum autoencoders. Quantum, 7, 942.", font_size=16).to_corner(DR)))
            ansatz = ImageMobject("assets/ansatz.png")
            self.play(FadeIn(ansatz))

            self.wait_until_bookmark("F")
            self.play(FadeOut(ansatz))
            anstanz1 = ImageMobject("assets/ansatz1.png")
            self.play(FadeIn(anstanz1))
            self.play(FadeOut(anstanz1))
            anstanz2 = ImageMobject("assets/ansatz2.png")
            self.play(FadeIn(anstanz2))
            self.play(FadeOut(anstanz2))
            anstanz3 = ImageMobject("assets/ansatz3.png")
            self.play(FadeIn(anstanz3))

            self.wait_until_bookmark("G")
            cost = MathTex(r"C = 1 - \frac{1}{N} \sum_{i=1}^N F(\rho^i_{\text{out}}, \rho^i_{\text{targ}})")
            cost.scale(0.6)
            cost.to_edge(UP)
            fidelity = MathTex(r"F(\rho_1, \rho_2) = \left( \operatorname{Tr} \sqrt{\sqrt{\rho_2} \rho_1 \sqrt{\rho_2}} \right)^2")
            fidelity.scale(0.6)
            fidelity.move_to(cost.get_center() + DOWN*0.7)
            self.play(Write(cost))
            self.play(Write(fidelity))

            self.wait_until_bookmark("H")
            self.play(FadeOut(anstanz3))
            self.play(FadeOut(cost))
            self.play(FadeOut(fidelity))
            ansatz4 = ImageMobject("assets/ansatz4.png")
            self.play(FadeIn(ansatz4))

            self.wait_until_bookmark("I")
            self.play(FadeOut(ansatz4))
            ansatz5 = ImageMobject("assets/ansatz5.png")
            self.play(FadeIn(ansatz5))

            self.wait_until_bookmark("J")
            self.play(FadeOut(ansatz5))
            flowchat = Flowchart()
            text = Text("Flowchart of COBYLA").to_edge(UP)
            self.play(AnimationGroup(Write(flowchat), Write(text)))

            self.wait_until_bookmark("K")
            self.play(FadeOut(flowchat))
            self.play(FadeOut(text))
            self.play(FadeOut(cite))

        with self.voiceover(text="With our Quantum Autoencoder set up with a forward pass, and optimization step, let's run it on MNIST digits! <bookmark mark='A'/> Training goes incredibly smoothly on both Domain Wall and MNIST datasets. <bookmark mark='B'/> So we have a simple quantum autoencoder working as well as a simple classical autoencoder, and it actually provides significant advantages due to the unitary encoder. So are we done? NOOOOO, we still have error correcting quantum autoencoders!! <bookmark mark='C'/>") as tracker:
            self.wait_until_bookmark("A")
            graph = ImageMobject("assets/training.png")
            self.play(FadeIn(graph))

            self.wait_until_bookmark("B")
            self.play(FadeOut(graph))
            graph = ImageMobject("assets/mnist.png")
            self.play(FadeIn(graph))

            self.wait_until_bookmark("C")
            self.play(FadeOut(graph))

        with self.voiceover(text="We need to make a couple of minor modifications to our current training circut to support error correction. <bookmark mark='A'/> Since we intend to change the data, we need to go back to training using <bookmark mark='B'/> fidelity instead of the SWAP test. <bookmark mark='C'/>We also need to encode our singular input qbit into a logical codespace, represented by |0_L⟩ = |000⟩ and |1_L⟩ = |111⟩. Finally, we include a block that introduces errors into the training. However, as a note, if we are training on real quantum hardware we don't need to arifically induce errors, as the autoencoder will simply train on the errors that occur the quantum hardware. This adaptibility makes quantum autoencoders an incredibly powerful error correcting tool. <bookmark mark='D'/> Running with a simulated p=0.05 bit flip error on each qbit, we get the following optimizer graph. By simply training on the |0_L⟩, |1_L⟩, and |+_L⟩ inputs, the Quantum Autoencoder learns perfect single qbit bit flip error correction, while still recovering the input state with when no errors occur. <bookmark mark='E'/>") as tracker:
            ansatz5 = ImageMobject("assets/ansatz5.png")
            self.play(FadeIn(ansatz5))

            self.wait_until_bookmark("A")
            self.play(FadeOut(ansatz5))
            ansatz6 = ImageMobject("assets/ansatz6.png")
            self.play(FadeIn(ansatz6))

            self.wait_until_bookmark("B")
            cost = MathTex(r"C = 1 - \frac{1}{N} \sum_{i=1}^N F(\rho^i_{\text{out}}, \rho^i_{\text{targ}})")
            cost.scale(0.6)
            cost.to_edge(DOWN)
            self.play(Write(cost))

            self.wait_until_bookmark("C")
            encoding = MathTex(r"\text{Encoding: } \begin{cases} |0_L\rangle = |000\rangle \\ |1_L\rangle = |111\rangle \end{cases}")
            encoding.scale(0.6)
            encoding.to_edge(UP)
            self.play(Write(encoding))

            self.wait_until_bookmark("D")
            self.play(FadeOut(ansatz6))
            self.play(FadeOut(encoding))
            self.play(FadeOut(cost))
            graph = ImageMobject("assets/QEC.png")
            self.play(FadeIn(graph))

            self.wait_until_bookmark("E")
            self.play(FadeOut(graph))

        with self.voiceover(text="As a recap, we created a quantum forward pass using parameterized rotation gates. Then we calculated a cost function using either the SWAP test or calculating the fidelity of states. Finally, we used COBYLA to optimize our parameters without knowing their derivative with respect to the cost function. With this quantum autoencoder, we were able to input qubits encoded into a logical codespace, and train the autoencoder to remove the errors and return the original state. We ran this all in qiskit simulation, but if this were running on a quantum computer, it could train on the real errors that happen in the qbits rather than our artificially incited ones. Please see \"Quantum Error Correction with Quantum Autoencoders\" for more details on the topic, but they have determined that this 3-1-3 autoencoder error correction is always more accurate than 3 qbit actively corrected encoding schemes when the noise inside of the error correcting network is low (this is because the increased number of gates in the autoencoder makes internal noise more significant).") as tracker:
            ansatz7 = ImageMobject("assets/ansatz7.png")
            self.play(FadeIn(ansatz7))




class Flowchart(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.construct_flowchart()

    def construct_flowchart(self):
        # Text elements
        init = MathTex(r"\text{Initialization}\\ \text{One set } \rho = \rho_{\text{beg}}\\ X_0 = X_{\text{init}}").scale(0.5).shift(UP*2)
        increment = MathTex(r"\text{Incrementation}\\ (X = X_0 + \rho)").scale(0.5)
        check_rho = MathTex(r"\rho \leq \rho_{\text{end}} ?").scale(0.5).shift(LEFT*3)
        end = MathTex(r"\text{End}").scale(0.5).shift(LEFT*5)
        compare_f = MathTex(r"\|F(X)\| \leq \|F(X_0)\| ?").scale(0.5).shift(DOWN*2)
        update_rho = MathTex(r"\rho = \frac{\rho}{2}").scale(0.5).shift(LEFT*3 + DOWN*2)
        reset_x = MathTex(r"X=X_0").scale(0.5).shift(RIGHT*2 + DOWN)

        # Surrounding rectangles
        rect_init = SurroundingRectangle(init, buff = .1, color = WHITE)
        rect_increment = SurroundingRectangle(increment, buff = .1, color = WHITE)
        rect_check_rho = SurroundingRectangle(check_rho, buff = .1, color = WHITE)
        rect_end = SurroundingRectangle(end, buff = .1, color = WHITE)
        rect_compare_f = SurroundingRectangle(compare_f, buff = .1, color = WHITE)
        rect_update_rho = SurroundingRectangle(update_rho, buff = .1, color = WHITE)
        rect_reset_x = SurroundingRectangle(reset_x, buff = .1, color = WHITE)

        # Arrows
        arrow_init_to_increment = Arrow(init.get_bottom(), increment.get_top(), buff=0.1)
        arrow_increment_to_check_rho = Arrow(check_rho.get_right(), increment.get_left(), buff=0.1)
        arrow_check_rho_to_update_rho = Arrow(update_rho.get_top(), check_rho.get_bottom(), buff=0.1)
        arrow_check_rho_to_end = Arrow(check_rho.get_left(), end.get_right(), buff=0.1)
        arrow_compare_f_to_reset_x = Arrow(compare_f.get_right(), reset_x.get_bottom(), buff=0.1, path_arc=90*DEGREES)
        arrow_reset_x_to_increment = Arrow(reset_x.get_top(), increment.get_right(), path_arc=90*DEGREES, buff=0.1)
        arrow_increment_to_compare_f = Arrow(increment.get_bottom(), compare_f.get_top(), buff=0.1)
        arrow_compare_f_to_update_rho = Arrow(compare_f.get_left(), update_rho.get_right(), buff=0.1)

        # Labels
        label_check_rho_to_update_rho = Text("No").scale(0.5).next_to(arrow_check_rho_to_update_rho, LEFT, buff=0.1)
        label_check_rho_to_end = Text("Yes").scale(0.5).next_to(arrow_check_rho_to_end, DOWN, buff=0.1)
        label_compare_f_to_reset_x = Text("No").scale(0.5).next_to(arrow_compare_f_to_reset_x, DR*0.1, buff=0.1)
        label_compare_f_to_update_rho = Text("Yes").scale(0.5).next_to(arrow_compare_f_to_update_rho, DOWN, buff=0.1)

        # Adding to VGroup
        self.add(init, increment, check_rho, end, compare_f, update_rho, reset_x)
        self.add(rect_init, rect_increment, rect_check_rho, rect_end, rect_compare_f, rect_update_rho, rect_reset_x)
        self.add(arrow_init_to_increment, arrow_increment_to_check_rho, arrow_check_rho_to_update_rho, arrow_check_rho_to_end, arrow_compare_f_to_reset_x, arrow_reset_x_to_increment, arrow_increment_to_compare_f, arrow_compare_f_to_update_rho)
        self.add(label_check_rho_to_update_rho, label_check_rho_to_end, label_compare_f_to_reset_x, label_compare_f_to_update_rho)
