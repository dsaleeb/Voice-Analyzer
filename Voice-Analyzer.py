import tkinter as tk
from tkinter import ttk, messagebox
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import queue
import scipy.signal
from datetime import datetime
from collections import Counter

class VoiceAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Voice Analyzer")
        self.root.geometry("800x600")
        
        # Audio parameters
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.UPDATE_INTERVAL = 50  # milliseconds
        
        # Noise threshold
        self.NOISE_THRESHOLD = 0.01
        
        # Store all frequencies for averaging
        self.all_frequencies = []
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
        # Get available input devices
        self.input_devices = self.get_input_devices()
        self.selected_device = tk.StringVar()
        
        # Data queue for thread-safe communication
        self.data_queue = queue.Queue()
        
        # Initialize frequency history and time points
        self.freq_history = np.zeros(100)
        self.time_points = np.arange(100)
        
        # Create GUI
        self.setup_gui()
        
    def get_input_devices(self):
        devices = []
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # Only input devices
                name = device_info['name']
                devices.append((name, i))
        return devices
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control frame (top)
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Device selection
        device_frame = ttk.Frame(control_frame)
        device_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(device_frame, text="Input Device:").grid(row=0, column=0, padx=5)
        self.device_menu = ttk.Combobox(
            device_frame,
            state='readonly',
            width=30
        )
        self.device_menu.grid(row=0, column=1, padx=5)
        
        # Populate device menu
        self.device_menu['values'] = [device[0] for device in self.input_devices]
        if self.input_devices:
            self.device_menu.set(self.input_devices[0][0])  # Select first device
        
        # Status frame
        status_frame = ttk.Frame(main_frame, padding="5")
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Recording button
        self.record_button = ttk.Button(
            status_frame, 
            text="Start Recording",
            command=self.toggle_recording
        )
        self.record_button.grid(row=0, column=0, padx=5)
        
        # Current frequency display
        self.freq_label = ttk.Label(
            status_frame, 
            text="Current Frequency: -- Hz",
            font=('Arial', 12)
        )
        self.freq_label.grid(row=0, column=1, padx=5)
        
        # Current voice type display
        self.voice_label = ttk.Label(
            status_frame, 
            text="Current Voice Type: --",
            font=('Arial', 12)
        )
        self.voice_label.grid(row=0, column=2, padx=5)
        
        # Average voice type display
        self.avg_voice_label = ttk.Label(
            status_frame, 
            text="Average Voice Type: --",
            font=('Arial', 12, 'bold')
        )
        self.avg_voice_label.grid(row=0, column=3, padx=5)
        
        # Statistics frame
        self.stats_frame = ttk.Frame(main_frame, padding="5")
        self.stats_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Statistics labels
        self.stats_label = ttk.Label(
            self.stats_frame,
            text="Recording Statistics: Not Available",
            font=('Arial', 11)
        )
        self.stats_label.grid(row=0, column=0, padx=5)
        
        # Setup matplotlib figure
        self.setup_plot()
        
    def setup_plot(self):
        # Create figure
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Set up the plot
        self.ax.set_ylim(50, 300)
        self.ax.set_xlim(0, 100)
        self.ax.set_title('Frequency over Time')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Frequency (Hz)')
        
        # Add colored background for male/female ranges
        self.ax.axhspan(85, 155, color='lightblue', alpha=0.3, label='Male Range')
        self.ax.axhspan(165, 255, color='pink', alpha=0.3, label='Female Range')
        self.ax.legend()
        
        # Create line object for the frequency plot
        self.line, = self.ax.plot(self.time_points, self.freq_history, 'k-', linewidth=1.5)
        
        # Make the plot expand with the window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights to make the plot expand
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def get_selected_device_index(self):
        selected_name = self.device_menu.get()
        for device in self.input_devices:
            if device[0] == selected_name:
                return device[1]
        return 0  # Default to first device if not found
            
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        try:
            # Clear previous recording data
            self.all_frequencies = []
            self.freq_history = np.zeros(100)
            self.line.set_ydata(self.freq_history)
            self.canvas.draw()
            self.avg_voice_label.config(text="Average Voice Type: --")
            self.stats_label.config(text="Recording Statistics: Not Available")
            
            # Get selected device index
            device_index = self.get_selected_device_index()
            
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback
            )
            
            self.is_recording = True
            self.record_button.config(text="Stop Recording")
            self.device_menu.config(state='disabled')  # Disable device selection while recording
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not start recording: {str(e)}")
            
    def stop_recording(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.is_recording = False
        self.record_button.config(text="Start Recording")
        self.device_menu.config(state='readonly')  # Re-enable device selection
        
        # Calculate and display averages
        self.calculate_average_voice_type()
        
   
    def calculate_average_voice_type(self):
        if not self.all_frequencies:
            self.avg_voice_label.config(text="Average Voice Type: No Data")
            return
            
        # Filter out None values and get valid frequencies
        valid_freqs = [f for f in self.all_frequencies if f is not None]
        
        if not valid_freqs:
            self.avg_voice_label.config(text="Average Voice Type: No Valid Data")
            return
            
        # Calculate average frequency
        avg_freq = np.mean(valid_freqs)
        
        # Count voice type occurrences
        voice_types = []
        for freq in valid_freqs:
            if 85 <= freq <= 155:
                voice_types.append("Male")
            elif 165 <= freq <= 255:
                voice_types.append("Female")
            else:
                voice_types.append("Androgynous")
                
        voice_count = Counter(voice_types)
        total_samples = len(voice_types)
        
        # Calculate percentages
        male_percent = (voice_count["Male"] / total_samples) * 100
        female_percent = (voice_count["Female"] / total_samples) * 100
        androgynous_percent = (voice_count["Androgynous"] / total_samples) * 100
        
        # Determine dominant voice type
        dominant_type = max(voice_count.items(), key=lambda x: x[1])[0]
        
        # Update labels with comprehensive statistics
        stats_text = (
            f"Total Samples: {total_samples}\n"
            f"Average Frequency: {avg_freq:.1f} Hz\n"
            f"Male: {male_percent:.1f}% | Female: {female_percent:.1f}% | Androgynous: {androgynous_percent:.1f}%"
        )
        
        self.avg_voice_label.config(text=f"Average Voice Type: {dominant_type}")
        self.stats_label.config(text=stats_text)
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.analyze_frequency(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def analyze_frequency(self, audio_data):
        try:
            # Check RMS amplitude
            rms = np.sqrt(np.mean(np.square(audio_data)))
            if rms < self.NOISE_THRESHOLD:
                self.data_queue.put(None)
                return
                
            # Apply Hanning window
            windowed_data = audio_data * scipy.signal.windows.hann(len(audio_data))
            
            # Compute FFT
            fft_data = np.fft.fft(windowed_data)
            freqs = np.fft.fftfreq(len(fft_data), 1.0/self.RATE)
            
            # Get magnitude spectrum
            magnitude_spectrum = np.abs(fft_data)
            
            # Find dominant frequency
            positive_mask = freqs > 0
            positive_freqs = freqs[positive_mask]
            positive_magnitudes = magnitude_spectrum[positive_mask]
            
            # Filter for human voice range (50-300 Hz)
            voice_mask = (positive_freqs >= 50) & (positive_freqs <= 300)
            voice_freqs = positive_freqs[voice_mask]
            voice_magnitudes = positive_magnitudes[voice_mask]
            
            if len(voice_magnitudes) > 0 and np.max(voice_magnitudes) > 1:
                dominant_freq = voice_freqs[np.argmax(voice_magnitudes)]
                self.data_queue.put(dominant_freq)
                self.all_frequencies.append(dominant_freq)  # Store frequency for averaging
            else:
                self.data_queue.put(None)
                
        except Exception as e:
            print(f"Error in frequency analysis: {str(e)}")
    
    def update_plot(self):
        if self.is_recording:
            try:
                while not self.data_queue.empty():
                    freq = self.data_queue.get_nowait()
                    
                    if freq is not None:
                        # Update frequency history
                        self.freq_history = np.roll(self.freq_history, -1)
                        self.freq_history[-1] = freq
                        
                        # Update labels
                        self.freq_label.config(text=f"Current Frequency: {freq:.1f} Hz")
                        
                        # Determine voice type
                        if 85 <= freq <= 155:
                            voice_type = "Male"
                        elif 165 <= freq <= 255:
                            voice_type = "Female"
                        else:
                            voice_type = "Androgynous"
                        
                        self.voice_label.config(text=f"Current Voice Type: {voice_type}")
                    else:
                        # No significant sound detected
                        self.freq_label.config(text="Current Frequency: -- Hz")
                        self.voice_label.config(text="Current Voice Type: --")
                        
                        # Clear the last point in history
                        self.freq_history = np.roll(self.freq_history, -1)
                        self.freq_history[-1] = np.nan
                
                # Update plot
                self.line.set_ydata(self.freq_history)
                self.canvas.draw()
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error updating plot: {str(e)}")
            
            # Schedule next update
            self.root.after(self.UPDATE_INTERVAL, self.update_plot)
    
    def run(self):
        """Start the application main loop"""
        self.root.mainloop()
        
    def cleanup(self):
        """Clean up resources when closing the application"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

if __name__ == "__main__":
    app = VoiceAnalyzer()
    try:
        app.run()
    finally:
        app.cleanup()

