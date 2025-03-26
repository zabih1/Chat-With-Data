/**
 * Speech to Text integration for Chat with Documents module
 * This file provides voice input functionality for the document chat interface
 */

// Initialize when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
  // Setup speech to text functionality for document chat
  initSpeechToTextForDocChat();
});

/**
 * Initialize speech to text functionality for document chat
 */
function initSpeechToTextForDocChat() {
  // First check if the document chat form exists
  const docChatForm = document.getElementById('docChatForm');
  if (!docChatForm) return;
  
  const docMessageInput = document.getElementById('docMessageInput');
  if (!docMessageInput) return;
  
  // Create and insert mic button before the send button
  const sendButton = docChatForm.querySelector('button[type="submit"]');
  if (!sendButton) return;
  
  // Create microphone button
  const micButton = document.createElement('button');
  micButton.type = 'button'; // Important: type="button" prevents form submission
  micButton.className = 'btn btn-outline-primary';
  micButton.id = 'docMicButton';
  micButton.innerHTML = '<i class="fas fa-microphone"></i>';
  micButton.title = 'Click to start voice input';
  
  // Insert before the send button
  docChatForm.insertBefore(micButton, sendButton);
  
  // Variables to manage recording state
  let mediaRecorder;
  let audioChunks = [];
  let isRecording = false;
  
  // Enable/disable mic button when document chat is enabled/disabled
  const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.attributeName === 'disabled') {
        micButton.disabled = docMessageInput.disabled;
      }
    });
  });
  
  observer.observe(docMessageInput, { attributes: true });
  micButton.disabled = docMessageInput.disabled;
  
  // Handle mic button click
  micButton.addEventListener('click', function(e) {
    // Prevent any form submission
    e.preventDefault();
    e.stopPropagation();
    
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });
  
  // Start recording function
  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Update button state
      micButton.innerHTML = '<i class="fas fa-stop"></i>';
      micButton.classList.remove('btn-outline-primary');
      micButton.classList.add('btn-danger');
      micButton.title = 'Click to stop recording';
      isRecording = true;
      
      // Show recording indicator in the input field
      const originalPlaceholder = docMessageInput.placeholder;
      docMessageInput.placeholder = '🔴 Recording... Click stop when finished';
      docMessageInput.dataset.originalPlaceholder = originalPlaceholder;
      
      // Initialize media recorder
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      
      // Handle data availability
      mediaRecorder.addEventListener('dataavailable', event => {
        audioChunks.push(event.data);
      });
      
      // Handle recording stop
      mediaRecorder.addEventListener('stop', async () => {
        // Create audio blob and upload it
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        await uploadAudio(audioBlob);
        
        // Stop all tracks in the stream to release the microphone
        stream.getTracks().forEach(track => track.stop());
      });
      
      // Start recording
      mediaRecorder.start();
      
      // Add a recording status message
      updateStatus('Recording... Speak clearly into your microphone', false);
      
    } catch (error) {
      console.error('Error starting recording:', error);
      updateStatus('Error accessing microphone: ' + error.message, false);
      resetRecordingState();
    }
  }
  
  // Stop recording function
  function stopRecording() {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      resetRecordingState();
      updateStatus('Processing your speech...', true);
    }
  }
  
  // Reset the recording UI state
  function resetRecordingState() {
    micButton.innerHTML = '<i class="fas fa-microphone"></i>';
    micButton.classList.remove('btn-danger');
    micButton.classList.add('btn-outline-primary');
    micButton.title = 'Click to start voice input';
    isRecording = false;
    
    // Restore original placeholder
    if (docMessageInput.dataset.originalPlaceholder) {
      docMessageInput.placeholder = docMessageInput.dataset.originalPlaceholder;
    }
  }
  
  // Upload and process audio
  async function uploadAudio(audioBlob) {
    try {
      // Create a file from the blob
      const fileName = 'speech_' + new Date().getTime() + '.webm';
      const audioFile = new File([audioBlob], fileName, {
        type: 'audio/webm'
      });
      
      // Prepare form data
      const formData = new FormData();
      formData.append('audio', audioFile);
      
      // Send to the server
      const response = await fetch('/api/speech-to-text', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (data.success && data.transcript) {
        // Update the input field with the transcript
        docMessageInput.value = data.transcript;
        updateStatus('Speech converted to text successfully', true);
        
        // Automatically submit the form to send the question to the LLM
        setTimeout(() => {
          // Create and dispatch a click event on the submit button
          sendButton.click();
        }, 500);
      } else {
        updateStatus(data.error || 'Failed to convert speech to text', false);
      }
    } catch (error) {
      console.error('Error uploading audio:', error);
      updateStatus('Error processing audio: ' + error.message, false);
    }
  }
  
  // Helper function to update status message
  function updateStatus(message, isSuccess = true) {
    const documentStatus = document.getElementById('documentStatus');
    if (documentStatus) {
      documentStatus.textContent = message;
      documentStatus.className = `small ${isSuccess ? 'text-success' : 'text-danger'} mt-2`;
    }
  }
}