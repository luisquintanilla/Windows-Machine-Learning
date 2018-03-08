﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Windows.Storage;
using Windows.UI;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;

using WinMLExplorer.MLModels;
using WinMLExplorer.ViewModels;

namespace WinMLExplorer
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        public MainViewModel ViewModel { get; set; }

        public MainPage()
        {
            this.ViewModel = new MainViewModel();

            this.ViewModel.CurrentModel = this.ViewModel.Models[0];

            this.InitializeComponent();

            this.modelComboBox.SelectedItem = this.ViewModel.CurrentModel;

            if (this.ViewModel.CameraNames.Count() > 0)
            {
                this.cameraSourceComboBox.SelectedItem = this.ViewModel.CameraNames[0];
            }
        }

        private async void OnCameraSourceSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            await this.RestartWebCameraAsync();
        }

        private async void OnDeviceToggleToggled(object sender, RoutedEventArgs e)
        {
            await this.ViewModel.CurrentModel.SetIsGpuValue(this.deviceToggle.IsOn);
        }

        private async void OnModelSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            // Clear existing results
            this.ViewModel.Results.Clear();

            this.ViewModel.CurrentModel = (WinMLModel)this.modelComboBox.SelectedItem;

            if (this.cameraControl.CameraStreamState == Windows.Media.Devices.CameraStreamState.Streaming)
            {
                await this.RestartWebCameraAsync();
            }
            else
            {
                // Hide UI elements
                this.durationTextBlock.Visibility = Visibility.Collapsed;
                this.cameraSource.Visibility = Visibility.Collapsed;
                this.resultsViewer.Visibility = Visibility.Collapsed;
                this.webCamHostGrid.Visibility = Visibility.Collapsed;
                this.imageHostGrid.Visibility = Visibility.Collapsed;

                // Show UI elements
                this.landingMessage.Visibility = Visibility.Visible;
            }
        }

        private async void OnImagePickerSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            StorageFile selectedImageFile = (StorageFile)this.imagePickerGridView.SelectedValue;

            this.imagePickerFlyout.Hide();

            if (selectedImageFile != null)
            {
                // Stop video stream
                await this.cameraControl.StopStreamAsync();

                // Hide UI elements
                this.durationTextBlock.Visibility = Visibility.Collapsed;
                this.cameraSource.Visibility = Visibility.Collapsed;
                this.landingMessage.Visibility = Visibility.Collapsed;
                this.resultsViewer.Visibility = Visibility.Collapsed;
                this.webCamHostGrid.Visibility = Visibility.Collapsed;

                // Process images
                this.imageHostGrid.Visibility = Visibility.Visible;
                await this.imageControl.UpdateImageAsync(selectedImageFile);

                // Show UI elements
                this.durationTextBlock.Visibility = Visibility.Visible;
                this.resultsViewer.Visibility = Visibility.Visible;
            }
        }

        private void OnPageSizeChanged(object sender, SizeChangedEventArgs e)
        {
            UpdateWebCamHostGridSize();
        }

        private async void OnWebCamButtonClicked(object sender, RoutedEventArgs e)
        {
            await StartWebCameraAsync();
        }

        private async Task RestartWebCameraAsync()
        {
            if (this.cameraSourceComboBox.SelectedItem == null)
            {
                return;
            }

            if (this.cameraControl.CameraStreamState == Windows.Media.Devices.CameraStreamState.Streaming)
            {
                await this.cameraControl.StopStreamAsync();
                await Task.Delay(1000);
                await this.cameraControl.StartStreamAsync(this.ViewModel, this.cameraSourceComboBox.SelectedItem.ToString());
            }
        }

        private async Task StartWebCameraAsync()
        {
            if (this.cameraSourceComboBox.SelectedItem == null)
            {
                return;
            }

            // Hide UI elements
            this.durationTextBlock.Visibility = Visibility.Collapsed;
            this.imageHostGrid.Visibility = Visibility.Collapsed;
            this.landingMessage.Visibility = Visibility.Collapsed;
            this.resultsViewer.Visibility = Visibility.Collapsed;

            // Start camera
            this.webCamHostGrid.Visibility = Visibility.Visible;
            await this.cameraControl.StartStreamAsync(this.ViewModel, this.cameraSourceComboBox.SelectedItem.ToString());
            await Task.Delay(250);

            // Show UI elements
            this.cameraSource.Visibility = Visibility.Visible;
            this.durationTextBlock.Visibility = Visibility.Visible;
            this.resultsViewer.Visibility = Visibility.Visible;

            UpdateWebCamHostGridSize();
        }

        private void UpdateWebCamHostGridSize()
        {
            this.webCamHostGrid.Height = this.webCamHostGrid.ActualWidth / (this.cameraControl.CameraAspectRatio != 0 ? this.cameraControl.CameraAspectRatio : 1.777777777777);
        }
    }
}
