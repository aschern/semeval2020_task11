<script>
    
function handleHighlightMouseOver(el) {
    $('[id='+el.getAttribute('id')+']').addClass('active');
  }

function handleHighlightMouseOut(el) {
    $('[id='+el.getAttribute('id')+']').removeClass('active');
}
  
</script>